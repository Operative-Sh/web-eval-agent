#!/usr/bin/env python3
"""
Browser utilities using OpenAI CUA (Computer-Using Agent) instead of browser-use.
This is a refactored version that replaces the browser-use agent with OpenAI's CUA.
"""

import asyncio
import json
import logging
import uuid
import warnings
import os
from typing import Dict, Any, List, Optional
from collections import deque
import base64
import traceback

# Import log server function
from .log_server import send_log, send_browser_view

# Import Playwright types
from playwright.async_api import (
    async_playwright,
    Error as PlaywrightError,
    Page as PlaywrightPage,
)

# Import CUA integration
from .cua_integration import OpenAICUA, CUAConfig

# Define the maximum number of logs/requests to keep
MAX_LOG_ENTRIES = 1000

# --- URL Filtering for Network Requests ---
def should_log_network_request(request) -> bool:
    """Determine if a network request should be logged based on its type and URL."""
    url = request.url
    if "/node_modules/" in url:
        return False

    # Only log XHR requests
    if request.resource_type != "xhr" and request.resource_type != "fetch":
        return False

    # Skip common static file types
    extensions_to_filter = [
        ".js", ".css", ".woff", ".woff2", ".ttf", ".eot", ".svg",
        ".png", ".jpg", ".jpeg", ".gif", ".ico", ".map",
    ]

    for ext in extensions_to_filter:
        if url.endswith(ext) or f"{ext}?" in url:
            return False

    return True


# --- Log Storage (Global within this module using deque) ---
console_log_storage: deque = deque(maxlen=MAX_LOG_ENTRIES)
network_request_storage: deque = deque(maxlen=MAX_LOG_ENTRIES)

# --- Screenshot Storage (Global within this module) ---
screenshot_storage: List[Dict[str, Any]] = []

# Global variables
active_cdp_session = None
active_screencast_running = False
browser_task_loop = None
screenshot_task = None


# --- Log Handlers ---
async def _handle_console_message(message):
    try:
        text = message.text
        log_entry = {
            "type": message.type,
            "text": text,
            "location": message.location,
            "timestamp": asyncio.get_event_loop().time(),
        }
        console_log_storage.append(log_entry)

        if hasattr(message, "failure") and message.failure:
            send_log(
                f"CONSOLE ERROR [{log_entry['type']}]: {log_entry['text']} - {message.failure}",
                "❌",
                log_type="console",
            )
        else:
            send_log(
                f"CONSOLE [{log_entry['type']}]: {log_entry['text']}",
                "🖥️",
                log_type="console",
            )
    except Exception as e:
        send_log(f"Error handling console message: {e}", "❌", log_type="status")


async def _handle_request(request):
    try:
        if not should_log_network_request(request):
            return

        try:
            headers = await request.all_headers()
        except PlaywrightError as e:
            headers = {"error": f"Req Header Error: {e}"}
        except Exception as e:
            headers = {"error": f"Unexpected Req Header Error: {e}"}

        post_data = None
        try:
            if request.post_data:
                post_data_buffer = await request.post_data_buffer()
                if post_data_buffer:
                    try:
                        post_data = post_data_buffer.decode("utf-8", errors="replace")
                    except Exception:
                        post_data = repr(post_data_buffer)
                else:
                    post_data = ""
            else:
                post_data = None
        except PlaywrightError as e:
            post_data = f"Post Data Error: {e}"
        except Exception as e:
            post_data = f"Unexpected Post Data Error: {e}"

        request_entry = {
            "url": request.url,
            "method": request.method,
            "headers": headers,
            "postData": post_data,
            "timestamp": asyncio.get_event_loop().time(),
            "resourceType": request.resource_type,
            "is_navigation": request.is_navigation_request(),
            "id": id(request),
        }
        network_request_storage.append(request_entry)
        send_log(
            f"NET REQ [{request_entry['method']}]: {request_entry['url']}",
            "➡️",
            log_type="network",
        )
    except Exception as e:
        url = request.url if request else "Unknown URL"
        send_log(
            f"Error handling request event for {url}: {e}", "❌", log_type="status"
        )


async def _handle_response(response):
    req_id = id(response.request)
    url = response.url

    if not should_log_network_request(response.request):
        return

    try:
        headers = await response.all_headers()
    except PlaywrightError as e:
        headers = {"error": f"Resp Header Error: {e}"}
    except Exception as e:
        headers = {"error": f"Unexpected Resp Header Error: {e}"}

    response_data = {
        "status": response.status,
        "statusText": response.status_text,
        "headers": headers,
        "timestamp": asyncio.get_event_loop().time(),
    }

    for req in network_request_storage:
        if req.get("id") == req_id and "response" not in req:
            req["response"] = response_data
            req["response_status"] = response.status
            req["response_timestamp"] = response_data["timestamp"]
            send_log(
                f"NET RESP [{response_data['status']}]: {req['url']}",
                "⬅️",
                log_type="network",
            )
            break


# Non-async wrapper functions for event listeners
def handle_console_message(message):
    asyncio.create_task(_handle_console_message(message))


def handle_request(request):
    asyncio.create_task(_handle_request(request))


def handle_response(response):
    asyncio.create_task(_handle_response(response))


def handle_request_failed(request):
    url = request.url if request else "Unknown URL"
    send_log(f"NET FAILED: {url}", "❌", log_type="network")


def handle_web_error(error):
    error_text = str(error) if error else "Unknown web error"
    send_log(f"WEB ERROR: {error_text}", "❌", log_type="console")


def handle_page_error(error):
    error_text = str(error) if error else "Unknown page error"
    send_log(f"PAGE ERROR: {error_text}", "❌", log_type="console")


# Helper function to get persisted browser state
def _get_persisted_state() -> Optional[str]:
    """Check for and return the path to persisted browser state if it exists."""
    state_file = os.path.expanduser("~/.operative/browser_state/state.json")
    return state_file if os.path.exists(state_file) else None


async def run_browser_task(
    task: str, tool_call_id: str = None, api_key: str = None, headless: bool = True
) -> Dict[str, Any]:
    """
    Run a task using OpenAI CUA (Computer-Using Agent), sending logs to the dashboard.

    Args:
        task: The task to run.
        tool_call_id: The tool call ID for API headers.
        api_key: The API key for authentication.
        headless: Whether to run the browser in headless mode.

    Returns:
        Dict[str, Any]: Result dictionary with 'result' and 'screenshots' keys.
    """
    global console_log_storage, network_request_storage, screenshot_storage
    global active_cdp_session, active_screencast_running, browser_task_loop, screenshot_task

    # Store the current asyncio loop for input handling
    browser_task_loop = asyncio.get_running_loop()

    # Clear storage for this run
    console_log_storage.clear()
    network_request_storage.clear()
    screenshot_storage.clear()

    # Local variables
    playwright = None
    browser = None
    context = None
    page = None
    cua = None

    # Configure logging suppression
    logging.basicConfig(level=logging.CRITICAL)
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        # Ensure Tool Call ID
        if tool_call_id is None:
            tool_call_id = str(uuid.uuid4())
            send_log(f"Generated tool_call_id: {tool_call_id}", "🆔", log_type="status")

        # Initialize Playwright
        playwright = await async_playwright().start()
        
        # Launch browser with CDP enabled
        browser = await playwright.chromium.launch(
            headless=headless,
            args=["--remote-debugging-port=9222"],
        )
        send_log(f"Browser launched (headless={headless})", "🎭", log_type="status")

        # Check for persisted browser state
        persisted_state = _get_persisted_state()
        if persisted_state:
            send_log(f"Loading persisted browser state from {persisted_state}", "💾", log_type="status")

        # Create context with persisted state if available
        context = await browser.new_context(storage_state=persisted_state)
        
        # Set up event listeners on context
        context.on("console", handle_console_message)
        context.on("request", handle_request)
        context.on("requestfailed", handle_request_failed)
        context.on("response", handle_response)
        context.on("weberror", handle_web_error)
        context.on("pageerror", handle_page_error)
        
        send_log("Browser context created with event listeners", "👂", log_type="status")

        # Create page
        page = await context.new_page()
        
        # Set up CDP session for screencasting
        try:
            cdp_session = await context.new_cdp_session(page)
            active_cdp_session = cdp_session
            
            # Set up screencast frame handler
            async def handle_screencast_frame(params):
                if "data" not in params or "sessionId" not in params:
                    return
                    
                try:
                    image_data = params["data"]
                    image_data_url = f"data:image/jpeg;base64,{image_data}"
                    
                    # Send to frontend
                    await send_browser_view(image_data_url)
                    
                    # Acknowledge the frame
                    await cdp_session.send(
                        "Page.screencastFrameAck",
                        {"sessionId": params["sessionId"]},
                    )
                except Exception:
                    pass
                    
            cdp_session.on("Page.screencastFrame", handle_screencast_frame)
            
            # Start screencast
            await cdp_session.send(
                "Page.startScreencast",
                {
                    "format": "jpeg",
                    "quality": 80,
                    "maxWidth": 1920,
                    "maxHeight": 1080,
                },
            )
            active_screencast_running = True
            send_log("CDP screencast started", "📹", log_type="status")
            
        except Exception as e:
            send_log(f"Failed to start CDP screencast: {e}", "❌", log_type="status")
            
        # Initialize CUA
        cua_config = CUAConfig(
            display_width=1920,
            display_height=1080,
            environment="browser",
            reasoning_summary="concise",
            max_iterations=30,
            timeout_seconds=300
        )
        
        cua = OpenAICUA(api_key=api_key, config=cua_config)
        await cua.initialize_browser(page)
        send_log("OpenAI CUA initialized", "🤖", log_type="status")
        
        # Extract URL from task if it contains navigation instructions
        initial_url = None
        if "http" in task.lower():
            # Try to extract URL from task
            import re
            url_pattern = r'https?://[^\s]+'
            urls = re.findall(url_pattern, task)
            if urls:
                initial_url = urls[0]
                send_log(f"Extracted URL from task: {initial_url}", "🔗", log_type="status")
        
        # Run the CUA task
        send_log(f"Starting CUA task: {task}", "🏃", log_type="agent")
        cua_result = await cua.run_task(task, initial_url=initial_url)
        send_log("CUA task completed", "🏁", log_type="agent")
        
        # Store CUA screenshots in our format
        screenshot_storage.extend(cua_result.get("screenshots", []))
        
        # Log screenshot information
        send_log(f"Captured {len(screenshot_storage)} screenshots during task", "📸", log_type="status")
        
        # Return results
        return {
            "result": cua_result.get("result", "Task completed"),
            "screenshots": screenshot_storage
        }
        
    except Exception as e:
        error_message = f"Error in run_browser_task: {e}\n{traceback.format_exc()}"
        send_log(error_message, "❌", log_type="status")
        return {
            "result": error_message,
            "screenshots": screenshot_storage
        }
        
    finally:
        # Cleanup
        if active_screencast_running and active_cdp_session:
            try:
                await active_cdp_session.send("Page.stopScreencast")
            except Exception:
                pass
            active_screencast_running = False
            
        if active_cdp_session:
            try:
                await active_cdp_session.detach()
            except Exception:
                pass
            active_cdp_session = None
            
        if page:
            try:
                await page.close()
            except Exception:
                pass
                
        if context:
            try:
                await context.close()
            except Exception:
                pass
                
        if browser:
            try:
                await browser.close()
            except Exception:
                pass
                
        if playwright:
            try:
                await playwright.stop()
            except Exception:
                pass
                
        send_log("Browser resources cleaned up", "🧹", log_type="status")


# Export other functions that might be needed
def get_browser_task_loop():
    """Get the asyncio loop used by run_browser_task."""
    global browser_task_loop
    return browser_task_loop


async def handle_browser_input(event_type: str, details: Dict) -> None:
    """Handle browser input events from the frontend."""
    global active_cdp_session, active_screencast_running

    if not active_cdp_session:
        send_log("Input error: No active CDP session", "❌", log_type="status")
        return

    if not active_screencast_running:
        send_log("Input error: Screencast not running", "❌", log_type="status")
        return

    # Note: In the CUA implementation, user inputs are not directly forwarded
    # since the CUA model controls the browser. This function is kept for compatibility
    # but may not be actively used when CUA is in control.
    send_log(f"User input received but CUA is in control: {event_type}", "ℹ️", log_type="status")


# Agent control functions (kept for compatibility but may not apply to CUA)
agent_instance = None

def pause_agent():
    """Pause the agent (not applicable to CUA)."""
    send_log("Pause not supported in CUA mode", "ℹ️", log_type="status")
    return False

def resume_agent():
    """Resume the agent (not applicable to CUA)."""
    send_log("Resume not supported in CUA mode", "ℹ️", log_type="status")
    return False

def stop_agent():
    """Stop the agent (not applicable to CUA)."""
    send_log("Stop not supported in CUA mode", "ℹ️", log_type="status")
    return False

def get_agent_state():
    """Get the agent state."""
    return {"paused": False, "stopped": False}