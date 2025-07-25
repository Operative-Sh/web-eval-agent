#!/usr/bin/env python3
"""
OpenAI Computer Use Agent implementation
Based on OpenAI's CUA sample app
"""

import os
import asyncio
import base64
import httpx
from typing import Dict, Any, Optional, List
from playwright.async_api import async_playwright, Page, Browser

from .log_server import send_log

class OpenAICUA:
    """OpenAI Computer Use Agent that runs the full CUA loop"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.openai_api_base = "https://api.openai.com/v1/responses"
        self.playwright = None
        self.browser = None
        self.page = None
        self.screenshots = []
        
    async def __aenter__(self):
        # Initialize Playwright browser
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=False,
            args=["--disable-extensions", "--disable-file-system"]
        )
        context = await self.browser.new_context()
        self.page = await context.new_page()
        await self.page.set_viewport_size({"width": 1024, "height": 768})
        
        send_log("OpenAI CUA browser initialized", "🌐", log_type="status")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
            
    def get_dimensions(self):
        return (1024, 768)
        
    def get_environment(self):
        return "browser"
        
    async def screenshot(self) -> str:
        """Take a screenshot and return base64 encoded string"""
        screenshot_bytes = await self.page.screenshot(type="png")
        return base64.b64encode(screenshot_bytes).decode("utf-8")
        
    async def click(self, x: int, y: int, button: str = "left"):
        """Execute click action"""
        await self.page.mouse.click(x, y, button=button)
        send_log(f"Clicked at ({x}, {y})", "👆", log_type="status")
        
    async def type(self, text: str):
        """Type text"""
        await self.page.keyboard.type(text)
        send_log(f"Typed: {text[:20]}...", "⌨️", log_type="status")
        
    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int):
        """Scroll action"""
        await self.page.mouse.move(x, y)
        await self.page.evaluate(f"window.scrollBy({scroll_x}, {scroll_y})")
        send_log(f"Scrolled by ({scroll_x}, {scroll_y})", "📜", log_type="status")
        
    async def keypress(self, keys: List[str]):
        """Press keys"""
        for key in keys:
            await self.page.keyboard.press(key)
        send_log(f"Pressed keys: {keys}", "⌨️", log_type="status")
        
    async def wait(self, ms: int = 1000):
        """Wait for specified milliseconds"""
        await asyncio.sleep(ms / 1000)
        
    async def move(self, x: int, y: int):
        """Move mouse to position"""
        await self.page.mouse.move(x, y)
        
    async def double_click(self, x: int, y: int):
        """Double click at position"""
        await self.page.mouse.dblclick(x, y)
        
    async def drag(self, path: List[Dict[str, int]]):
        """Drag from start to end of path"""
        if not path:
            return
        await self.page.mouse.move(path[0]["x"], path[0]["y"])
        await self.page.mouse.down()
        for point in path[1:]:
            await self.page.mouse.move(point["x"], point["y"])
        await self.page.mouse.up()
        
    def get_current_url(self) -> str:
        """Get current page URL"""
        return self.page.url
        
    async def handle_computer_action(self, action: Dict[str, Any]):
        """Handle a computer action from CUA"""
        action_type = action.get("type")
        
        if action_type == "click":
            await self.click(action.get("x", 0), action.get("y", 0), action.get("button", "left"))
        elif action_type == "type":
            await self.type(action.get("text", ""))
        elif action_type == "scroll":
            await self.scroll(
                action.get("x", 0), 
                action.get("y", 0),
                action.get("scroll_x", 0),
                action.get("scroll_y", 0)
            )
        elif action_type == "keypress":
            await self.keypress(action.get("keys", []))
        elif action_type == "wait":
            await self.wait(action.get("ms", 1000))
        elif action_type == "move":
            await self.move(action.get("x", 0), action.get("y", 0))
        elif action_type == "double_click":
            await self.double_click(action.get("x", 0), action.get("y", 0))
        elif action_type == "drag":
            await self.drag(action.get("path", []))
        elif action_type == "screenshot":
            pass  # Screenshot is taken after every action
        else:
            send_log(f"Unknown action type: {action_type}", "❓", log_type="status")
            
    async def run_task(self, task: str) -> Dict[str, Any]:
        """Run a task using OpenAI CUA"""
        send_log(f"Starting OpenAI CUA task: {task}", "🤖", log_type="agent")
        
        # Initial screenshot
        initial_screenshot = await self.screenshot()
        
        # Prepare initial request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Add OpenAI organization if set
        openai_org = os.getenv("OPENAI_ORG")
        if openai_org:
            headers["Openai-Organization"] = openai_org
        
        request_data = {
            "model": "computer-use-preview",
            "tools": [{
                "type": "computer-preview",
                "display_width": 1024,
                "display_height": 768,
                "environment": "browser"
            }],
            "input": [{
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": task
                }, {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{initial_screenshot}"
                }]
            }],
            "reasoning": {
                "summary": "concise"
            },
            "truncation": "auto"
        }
        
        items = []
        step_count = 0
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            while True:
                # Make API request
                response = await client.post(
                    self.openai_api_base,
                    headers=headers,
                    json=request_data
                )
                
                if response.status_code != 200:
                    error_msg = f"API error: {response.status_code} - {response.text}"
                    send_log(error_msg, "❌", log_type="status")
                    return {"error": error_msg}
                    
                response_data = response.json()
                
                # Process output items
                output_items = response_data.get("output", [])
                items.extend(output_items)
                
                # Look for computer calls
                computer_calls = [
                    item for item in output_items 
                    if item.get("type") == "computer_call"
                ]
                
                # If no computer calls, we're done
                if not computer_calls:
                    # Extract final message
                    for item in output_items:
                        if item.get("type") == "message":
                            content = item.get("content", [])
                            if content and content[0].get("type") == "text":
                                send_log(content[0].get("text", ""), "💬", log_type="agent")
                    break
                    
                # Process computer calls
                for computer_call in computer_calls:
                    step_count += 1
                    action = computer_call.get("action", {})
                    call_id = computer_call.get("call_id")
                    
                    # Log the action
                    send_log(f"Step {step_count}: {action.get('type')}({action})", "📍", log_type="agent")
                    
                    # Execute the action
                    await self.handle_computer_action(action)
                    
                    # Take screenshot after action
                    screenshot_base64 = await self.screenshot()
                    
                    # Store screenshot
                    self.screenshots.append({
                        "step": step_count,
                        "url": self.page.url,
                        "screenshot": screenshot_base64,
                        "timestamp": asyncio.get_event_loop().time()
                    })
                    
                    # Handle safety checks
                    pending_checks = computer_call.get("pending_safety_checks", [])
                    
                    # Prepare next request with screenshot
                    request_data = {
                        "model": "computer-use-preview",
                        "previous_response_id": response_data["id"],
                        "tools": [{
                            "type": "computer-preview",
                            "display_width": 1024,
                            "display_height": 768,
                            "environment": "browser"
                        }],
                        "input": [{
                            "call_id": call_id,
                            "type": "computer_call_output",
                            "acknowledged_safety_checks": pending_checks,
                            "output": {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{screenshot_base64}"
                            },
                            "current_url": self.page.url
                        }],
                        "truncation": "auto"
                    }
                    
        send_log("OpenAI CUA task completed", "✅", log_type="agent")
        
        # Extract final text from all messages
        final_output = []
        for item in items:
            if item.get("type") == "message":
                content = item.get("content", [])
                for c in content:
                    if c.get("type") == "text":
                        final_output.append(c.get("text", ""))
                        
        return {
            "result": "\n".join(final_output),
            "screenshots": self.screenshots
        }


async def run_openai_cua_task(task: str, api_key: str) -> Dict[str, Any]:
    """Run a task using OpenAI Computer Use API"""
    
    async with OpenAICUA(api_key) as cua:
        try:
            # Navigate to initial URL if provided in task
            if "VISIT:" in task:
                url_match = task.split("VISIT:")[1].split("GOAL:")[0].strip()
                await cua.page.goto(url_match)
                send_log(f"Navigated to: {url_match}", "🔗", log_type="status")
                
            result = await cua.run_task(task)
            return result
            
        except Exception as e:
            import traceback
            error_msg = f"Error in OpenAI CUA: {e}\n{traceback.format_exc()}"
            send_log(error_msg, "❌", log_type="status")
            return {"error": error_msg, "screenshots": []}