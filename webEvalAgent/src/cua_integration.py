#!/usr/bin/env python3
"""
OpenAI Computer-Using Agent (CUA) Integration Module

This module provides integration with OpenAI's computer-use-preview model
for web browser automation and evaluation tasks.
"""

import asyncio
import base64
import json
import traceback
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import httpx
from playwright.async_api import Page as PlaywrightPage, CDPSession

from .log_server import send_log
from .env_utils import get_backend_url


@dataclass
class CUAConfig:
    """Configuration for the Computer-Using Agent."""
    model: str = "computer-use-preview"
    display_width: int = 1024
    display_height: int = 768
    environment: str = "browser"
    reasoning_summary: str = "concise"
    truncation: str = "auto"
    max_iterations: int = 50
    timeout_seconds: int = 300


@dataclass
class CUAState:
    """State tracking for the CUA session."""
    page: Optional[PlaywrightPage] = None
    cdp_session: Optional[CDPSession] = None
    previous_response_id: Optional[str] = None
    screenshots: List[Dict[str, Any]] = field(default_factory=list)
    completed: bool = False
    error: Optional[str] = None
    iteration_count: int = 0


class OpenAICUA:
    """OpenAI Computer-Using Agent implementation for browser automation."""
    
    def __init__(self, api_key: str, config: Optional[CUAConfig] = None, tool_call_id: Optional[str] = None):
        """Initialize the CUA with API key and configuration.
        
        Args:
            api_key: Operative API key (not OpenAI API key)
            config: Optional CUA configuration
            tool_call_id: Tool call ID for tracking
        """
        self.api_key = api_key
        self.config = config or CUAConfig()
        self.state = CUAState()
        self.tool_call_id = tool_call_id
        
        # Get backend URL from environment
        self.backend_url = get_backend_url("v1beta/openai/cua")
        
    async def initialize_browser(self, page: PlaywrightPage) -> None:
        """Initialize browser page and CDP session for CUA.
        
        Args:
            page: Playwright page instance
        """
        self.state.page = page
        
        # Set up CDP session for screenshots
        try:
            self.state.cdp_session = await page.context.new_cdp_session(page)
            send_log("CDP session initialized for CUA", "🔧", log_type="status")
        except Exception as e:
            send_log(f"Failed to initialize CDP session: {e}", "❌", log_type="status")
            raise
            
    async def capture_screenshot(self) -> str:
        """Capture a screenshot of the current page state.
        
        Returns:
            Base64 encoded screenshot data
        """
        if not self.state.page:
            raise ValueError("Browser page not initialized")
            
        try:
            # Use Playwright's screenshot method
            screenshot_bytes = await self.state.page.screenshot(type="jpeg", quality=90)
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            return screenshot_base64
        except Exception as e:
            send_log(f"Failed to capture screenshot: {e}", "❌", log_type="status")
            raise
            
    async def execute_action(self, action: Dict[str, Any]) -> None:
        """Execute a CUA action on the browser.
        
        Args:
            action: The action dictionary from CUA response
        """
        if not self.state.page:
            raise ValueError("Browser page not initialized")
            
        action_type = action.get("type")
        
        try:
            if action_type == "click":
                x = action.get("x", 0)
                y = action.get("y", 0)
                button = action.get("button", "left")
                
                await self.state.page.mouse.click(x, y, button=button)
                send_log(f"Clicked at ({x}, {y}) with {button} button", "🖱️", log_type="agent")
                
            elif action_type == "type":
                text = action.get("text", "")
                await self.state.page.keyboard.type(text)
                send_log(f"Typed: {text[:50]}{'...' if len(text) > 50 else ''}", "⌨️", log_type="agent")
                
            elif action_type == "keypress":
                keys = action.get("keys", [])
                for key in keys:
                    # Map common key names
                    if key.upper() == "ENTER":
                        await self.state.page.keyboard.press("Enter")
                    elif key.upper() == "SPACE":
                        await self.state.page.keyboard.press(" ")
                    elif key.upper() == "TAB":
                        await self.state.page.keyboard.press("Tab")
                    elif key.upper() == "ESCAPE":
                        await self.state.page.keyboard.press("Escape")
                    else:
                        await self.state.page.keyboard.press(key)
                send_log(f"Pressed keys: {', '.join(keys)}", "⌨️", log_type="agent")
                
            elif action_type == "scroll":
                x = action.get("x", 0)
                y = action.get("y", 0)
                scroll_x = action.get("scrollX", 0)
                scroll_y = action.get("scrollY", 0)
                
                await self.state.page.mouse.move(x, y)
                await self.state.page.evaluate(f"window.scrollBy({scroll_x}, {scroll_y})")
                send_log(f"Scrolled by ({scroll_x}, {scroll_y}) at ({x}, {y})", "📜", log_type="agent")
                
            elif action_type == "wait":
                # Default wait time of 2 seconds
                await asyncio.sleep(2)
                send_log("Waited for page changes", "⏳", log_type="agent")
                
            elif action_type == "screenshot":
                # Just a marker action, screenshot is taken after each action anyway
                send_log("Screenshot requested", "📸", log_type="agent")
                
            else:
                send_log(f"Unknown action type: {action_type}", "❓", log_type="status")
                
        except Exception as e:
            error_msg = f"Failed to execute {action_type} action: {e}"
            send_log(error_msg, "❌", log_type="status")
            raise
            
    async def run_task(self, task: str, initial_url: Optional[str] = None) -> Dict[str, Any]:
        """Run a task using the Computer-Using Agent.
        
        Args:
            task: The task description to execute
            initial_url: Optional URL to navigate to before starting
            
        Returns:
            Dictionary containing the result and screenshots
        """
        if not self.state.page:
            raise ValueError("Browser not initialized. Call initialize_browser first.")
            
        # Navigate to initial URL if provided
        if initial_url:
            await self.state.page.goto(initial_url)
            send_log(f"Navigated to {initial_url}", "🌐", log_type="agent")
            
        # Capture initial screenshot
        initial_screenshot = await self.capture_screenshot()
        
        # Create initial request
        input_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": task
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{initial_screenshot}"
                    }
                ]
            }
        ]
        
        send_log(f"Starting CUA task: {task}", "🚀", log_type="agent")
        
        try:
            # Initial request to CUA via Operative backend
            request_data = {
                "model": self.config.model,
                "tools": [{
                    "type": "computer_use_preview",
                    "display_width": self.config.display_width,
                    "display_height": self.config.display_height,
                    "environment": self.config.environment
                }],
                "input": input_messages,
                "reasoning": {
                    "summary": self.config.reasoning_summary
                },
                "truncation": self.config.truncation
            }
            
            headers = {
                "x-operative-api-key": self.api_key,
                "x-operative-tool-call-id": self.tool_call_id or "unknown",
                "x-operative-tool-name": "web_eval_agent",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{self.backend_url}/v1/responses",
                    json=request_data,
                    headers=headers
                )
                resp.raise_for_status()
                response = resp.json()
            
            self.state.previous_response_id = response.get('id')
            
            # Process the response and execute actions in a loop
            while not self.state.completed and self.state.iteration_count < self.config.max_iterations:
                self.state.iteration_count += 1
                
                # Process output items
                computer_calls = []
                for item in response.get('output', []):
                    item_type = item.get('type')
                    if item_type == "reasoning":
                        # Log reasoning summary
                        summary = item.get('summary', [])
                        if summary:
                            for summary_item in summary:
                                if isinstance(summary_item, dict) and 'text' in summary_item:
                                    send_log(f"CUA reasoning: {summary_item['text']}", "🤔", log_type="agent")
                                    
                    elif item_type == "computer_call":
                        computer_calls.append(item)
                        
                    elif item_type == "text":
                        # Log any text output from the model
                        if 'text' in item:
                            send_log(f"CUA message: {item['text']}", "💬", log_type="agent")
                
                # If no computer calls, we're done
                if not computer_calls:
                    send_log("CUA task completed - no more actions", "✅", log_type="agent")
                    self.state.completed = True
                    break
                    
                # Execute the computer call (should be only one per response)
                for computer_call in computer_calls:
                    # Check for pending safety checks
                    pending_checks = computer_call.get('pending_safety_checks', [])
                    if pending_checks:
                        # Log safety checks but acknowledge them (in production, you'd want user confirmation)
                        for check in pending_checks:
                            send_log(f"Safety check: {check.get('message', 'Unknown check')}", "⚠️", log_type="status")
                        
                    # Execute the action
                    action = computer_call.get('action')
                    if action:
                        await self.execute_action(action)
                        
                    # Wait a bit for the action to take effect
                    await asyncio.sleep(1)
                    
                    # Capture screenshot after action
                    screenshot = await self.capture_screenshot()
                    
                    # Store screenshot
                    self.state.screenshots.append({
                        "step": self.state.iteration_count,
                        "screenshot": screenshot,
                        "action": computer_call.get('action')
                    })
                    
                    # Send screenshot back to CUA via Operative backend
                    request_data = {
                        "model": self.config.model,
                        "previous_response_id": self.state.previous_response_id,
                        "tools": [{
                            "type": "computer_use_preview",
                            "display_width": self.config.display_width,
                            "display_height": self.config.display_height,
                            "environment": self.config.environment
                        }],
                        "input": [{
                            "call_id": computer_call.get('call_id'),
                            "type": "computer_call_output",
                            "output": {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{screenshot}"
                            },
                            # Include current URL if available
                            "current_url": self.state.page.url if self.state.page else None
                        }],
                        "truncation": self.config.truncation
                    }
                    
                    async with httpx.AsyncClient(timeout=300.0) as client:
                        resp = await client.post(
                            f"{self.backend_url}/v1/responses",
                            json=request_data,
                            headers=headers
                        )
                        resp.raise_for_status()
                        response = resp.json()
                    
                    self.state.previous_response_id = response.get('id')
                    
            # Return results
            return {
                "result": "Task completed successfully" if self.state.completed else f"Task stopped after {self.state.iteration_count} iterations",
                "screenshots": self.state.screenshots,
                "iterations": self.state.iteration_count,
                "completed": self.state.completed,
                "error": self.state.error
            }
            
        except Exception as e:
            error_msg = f"CUA task failed: {str(e)}\n{traceback.format_exc()}"
            send_log(error_msg, "❌", log_type="status")
            self.state.error = str(e)
            
            return {
                "result": f"Error: {str(e)}",
                "screenshots": self.state.screenshots,
                "iterations": self.state.iteration_count,
                "completed": False,
                "error": str(e)
            }