#!/usr/bin/env python3
# /// script
# dependencies = [
#   "mcp>=0.9.0",
#   "python-dotenv>=1.0.0",
#   "requests>=2.31.0",
#   "websocket-client>=1.6.0",
#   "Pillow>=10.0.0",
# ]
# ///
"""
MCP Image Generator Server

Provides image generation services using ComfyUI for The Ephergent story
generation workflow.

Exposes:
- 4 image generation tools
- 3 MCP resources for accessing generated images

Usage:
    uv run server.py
    python server.py

Configuration via Claude Desktop/Code:
    {
      "mcpServers": {
        "ephergent-image-generator": {
          "command": "uv",
          "args": ["run", "/absolute/path/to/MCP_image_generator/server.py"],
          "env": {
            "COMFYUI_URL": "http://comfyui.nexus.home.test",
            "IMAGE_OUTPUT_DIR": "/path/to/output/dir"
          }
        }
      }
    }
"""

import sys
import os
import json
import logging
import time
import uuid
import random
import requests
import websocket
from pathlib import Path
from typing import Any, Sequence, Optional, Dict, List
from urllib.parse import urlencode

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

from shared import BaseMCPServer, ConfigError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image_generator")


class ComfyUIClient:
    """Client for interacting with ComfyUI API."""

    def __init__(self, comfyui_url: str):
        """Initialize ComfyUI client."""
        self.comfyui_url = comfyui_url

        # Load workflow template
        assets_dir = Path(__file__).parent.parent / 'assets'
        self.workflow_path = assets_dir / 'workflows' / 't2i_ephergent_season_03_workflow.json'

        # Style constants
        self.style_prefix = "ArsMJStyle, dnddarkestfantasy, Kenva, fluxlisimo, fluxlisimo_neon, CCM-R-Daal, "
        self.post_style_suffix = """A digitally illustrated drawing in stylized 3D anime manga style with painterly cel-shading and hand-drawn textures, featuring volumetric lighting with soft watercolor-like gradients, dynamic comic book halftone patterns, realistic depth of field and atmospheric perspective, soft ambient occlusion shadows, cinematic rim lighting, subsurface scattering effects, realistic material textures and fabric physics, while maintaining clean manga lineart with NPR non-photorealistic rendering and traditional anime color palettes enhanced by atmospheric haze"""

        logger.info(f"ComfyUI Client initialized - URL: {self.comfyui_url}")

        if not self.workflow_path.exists():
            logger.error(f"ComfyUI workflow template not found: {self.workflow_path}")
            raise FileNotFoundError(f"Workflow template not found: {self.workflow_path}")

    def is_available(self) -> bool:
        """Check if ComfyUI is available."""
        try:
            response = requests.get(f"{self.comfyui_url}/system_stats", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"ComfyUI not available: {e}")
            return False

    def test_connection(self) -> Dict[str, Any]:
        """Test ComfyUI connection and return detailed status."""
        status = {
            'available': False,
            'url': self.comfyui_url,
            'error': None,
            'response_time': None
        }

        try:
            start_time = time.time()
            response = requests.get(f"{self.comfyui_url}/system_stats", timeout=10)
            response_time = time.time() - start_time

            status['response_time'] = round(response_time, 3)

            if response.status_code == 200:
                status['available'] = True
                logger.info(f"ComfyUI connection test successful - {response_time:.3f}s response time")
            else:
                status['error'] = f"HTTP {response.status_code}"

        except requests.exceptions.ConnectionError:
            status['error'] = 'Connection refused - server offline'
        except requests.exceptions.Timeout:
            status['error'] = 'Connection timeout - server overloaded'
        except Exception as e:
            status['error'] = str(e)

        return status

    def _load_workflow_template(self) -> Dict:
        """Load and prepare the workflow template."""
        try:
            with open(self.workflow_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
            logger.info(f"Loaded ComfyUI workflow template: {self.workflow_path}")
            return workflow
        except Exception as e:
            logger.error(f"Error loading workflow template: {e}")
            raise

    def _update_prompt_in_workflow(self, workflow: Dict, prompt: str) -> Dict:
        """Update the prompt text in the workflow."""
        try:
            # Find positive prompt node IDs by looking at KSampler connections
            positive_prompt_node_ids = set()
            ksampler_nodes = {
                nid: ndata for nid, ndata in workflow.items()
                if "KSampler" in ndata.get("class_type", "")
            }

            for ksampler_data in ksampler_nodes.values():
                pos_input = ksampler_data.get("inputs", {}).get("positive")
                if isinstance(pos_input, list) and len(pos_input) > 0:
                    positive_prompt_node_ids.add(pos_input[0])

            # Update prompt text in positive prompt nodes
            for node_id in positive_prompt_node_ids:
                if node_id in workflow and 'inputs' in workflow[node_id]:
                    # Combine style prefix + prompt + style suffix
                    full_prompt = f"{self.style_prefix}{prompt}, {self.post_style_suffix}"
                    workflow[node_id]['inputs']['text'] = full_prompt
                    logger.info(f"Updated prompt in node {node_id}")

            # Update random seed for variation
            seed = random.randint(10**14, 10**15 - 1)
            for node_id in ksampler_nodes:
                if 'inputs' in workflow[node_id]:
                    workflow[node_id]['inputs']['seed'] = seed

            logger.info(f"Updated workflow with prompt and seed {seed}")
            return workflow

        except Exception as e:
            logger.error(f"Error updating workflow prompt: {e}")
            raise

    def _open_websocket_connection(self) -> tuple[Optional[websocket.WebSocket], Optional[str]]:
        """Open WebSocket connection to ComfyUI."""
        client_id = str(uuid.uuid4())
        ws_url = f"ws://{self.comfyui_url.replace('http://', '').replace('https://', '')}/ws?clientId={client_id}"

        try:
            ws = websocket.create_connection(ws_url, timeout=15)
            logger.info(f"WebSocket connection established: {ws_url}")
            return ws, client_id
        except Exception as e:
            logger.error(f"Failed to connect to ComfyUI WebSocket: {e}")
            return None, None

    def _queue_prompt(self, prompt_workflow: dict, client_id: str) -> Optional[str]:
        """Queue a prompt for generation."""
        payload = {"prompt": prompt_workflow, "client_id": client_id}
        data = json.dumps(payload).encode('utf-8')
        url = f"{self.comfyui_url}/prompt"
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

        try:
            response = requests.post(url, data=data, headers=headers, timeout=30)
            response.raise_for_status()
            response_data = response.json()
            prompt_id = response_data['prompt_id']
            logger.info(f"Queued prompt with ID: {prompt_id}")
            return prompt_id
        except Exception as e:
            logger.error(f"Failed to queue prompt: {e}")
            return None

    def _track_progress(self, prompt_workflow: dict, ws: websocket.WebSocket, prompt_id: str) -> Dict[str, Any]:
        """Track progress of image generation via WebSocket."""
        save_node_ids = {
            nid for nid, node_info in prompt_workflow.items()
            if node_info.get("class_type", "").startswith("SaveImage")
        }

        timeout_start = time.time()
        timeout_duration = 300  # 5 minutes timeout
        progress_info = {'completed': False, 'progress': 0, 'max': 0}

        while time.time() - timeout_start < timeout_duration:
            try:
                ws.settimeout(10)
                out = ws.recv()

                if isinstance(out, str):
                    message = json.loads(out)

                    if message.get('type') == 'executed' and message.get('data', {}).get('prompt_id') == prompt_id:
                        node_id = message['data'].get('node')
                        if node_id in save_node_ids:
                            logger.info(f"Save node {node_id} executed. Prompt {prompt_id} complete.")
                            progress_info['completed'] = True
                            return progress_info

                    elif message.get('type') == 'progress':
                        data = message.get('data', {})
                        value = data.get('value', 0)
                        max_value = data.get('max', 0)
                        progress_info['progress'] = value
                        progress_info['max'] = max_value
                        if max_value > 0:
                            progress_pct = (value / max_value) * 100
                            logger.info(f"Generation progress: {progress_pct:.1f}%")

            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:
                logger.error(f"Error during progress tracking: {e}")
                break

        logger.warning(f"Timeout waiting for prompt {prompt_id} completion")
        return progress_info

    def _get_image(self, filename: str, subfolder: str, folder_type: str) -> Optional[bytes]:
        """Download image from ComfyUI."""
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url = f"{self.comfyui_url}/view?{urlencode(params)}"

        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            logger.info(f"Successfully downloaded image '{filename}'")
            return response.content
        except Exception as e:
            logger.error(f"Failed to fetch image {filename}: {e}")
            return None

    def _get_history(self, prompt_id: str) -> Optional[dict]:
        """Get generation history from ComfyUI."""
        url = f"{self.comfyui_url}/history/{prompt_id}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch history for {prompt_id}: {e}")
            return None

    def _get_images_from_history(self, prompt_id: str) -> List[bytes]:
        """Extract image data from generation history."""
        history_data = self._get_history(prompt_id)
        if not history_data or prompt_id not in history_data:
            return []

        output_images_data = []
        for node_id, node_output in history_data[prompt_id]['outputs'].items():
            if 'images' in node_output:
                for image_info in node_output['images']:
                    if image_info.get('type') == 'output':
                        image_data = self._get_image(
                            image_info['filename'],
                            image_info.get('subfolder', ''),
                            image_info['type']
                        )
                        if image_data:
                            output_images_data.append(image_data)

        return output_images_data

    def generate_image(self, prompt: str, output_path: Path, width: int = 1344, height: int = 768) -> Optional[Path]:
        """
        Generate an image using ComfyUI.

        Args:
            prompt: Text prompt for image generation
            output_path: Where to save the generated image
            width: Image width (default: 1344)
            height: Image height (default: 768)

        Returns:
            Path to generated image or None if failed
        """
        if not self.is_available():
            logger.error("ComfyUI service not available")
            return None

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load workflow template
        workflow = self._load_workflow_template()

        # Set custom dimensions
        for node in workflow.values():
            if node.get('class_type', '').startswith('EmptyLatentImage'):
                node['inputs']['width'] = width
                node['inputs']['height'] = height
                logger.info(f"Set custom dimensions: {width}x{height}")

        # Update prompt in workflow
        workflow = self._update_prompt_in_workflow(workflow, prompt)

        # Open WebSocket connection
        ws, client_id = self._open_websocket_connection()
        if not ws:
            return None

        try:
            # Queue the prompt
            prompt_id = self._queue_prompt(workflow, client_id)
            if not prompt_id:
                return None

            logger.info(f"Starting image generation (estimated ~200 seconds)...")

            # Track progress
            self._track_progress(workflow, ws, prompt_id)

            # Get generated images
            images_data = self._get_images_from_history(prompt_id)

            if images_data:
                # Save the first image
                with open(output_path, 'wb') as f:
                    f.write(images_data[0])
                logger.info(f"Image successfully saved to: {output_path}")
                return output_path
            else:
                logger.error(f"No images found in history for prompt ID: {prompt_id}")
                return None

        except Exception as e:
            logger.error(f"Error during image generation: {e}")
            return None
        finally:
            # Clean up WebSocket connection
            try:
                if ws and ws.connected:
                    ws.close()
            except:
                pass


class ImageGeneratorServer(BaseMCPServer):
    """MCP server for image generation using ComfyUI."""

    def __init__(self):
        """Initialize the image generator server."""
        super().__init__(log_level="INFO")
        self.app = Server("ephergent-image-generator")

        # Get ComfyUI configuration
        self.comfyui_url = os.getenv('COMFYUI_URL', 'http://127.0.0.1:8188')
        self.comfyui_enabled = os.getenv('COMFYUI_ENABLED', 'true').lower() == 'true'

        # Get output directory
        self.output_dir = Path(os.getenv('IMAGE_OUTPUT_DIR', '/tmp/ephergent_images'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ComfyUI client
        self.comfyui_client: Optional[ComfyUIClient] = None
        if self.comfyui_enabled:
            try:
                self.comfyui_client = ComfyUIClient(self.comfyui_url)
            except Exception as e:
                logger.error(f"Failed to initialize ComfyUI client: {e}")
                self.comfyui_enabled = False

    def get_server_name(self) -> str:
        """Get server name."""
        return "image_generator"

    def validate_configuration(self) -> dict[str, Any]:
        """Validate server configuration."""
        errors = []
        warnings = []

        # Check ComfyUI URL
        if not self.comfyui_url:
            errors.append("COMFYUI_URL environment variable not set")

        # Check workflow template
        assets_dir = Path(__file__).parent.parent / 'assets'
        workflow_path = assets_dir / 'workflows' / 't2i_ephergent_season_03_workflow.json'
        if not workflow_path.exists():
            errors.append(f"ComfyUI workflow template not found: {workflow_path}")

        # Check output directory
        if not self.output_dir.exists():
            warnings.append(f"Output directory does not exist, will be created: {self.output_dir}")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    async def check_health(self) -> dict[str, Any]:
        """Check server health."""
        services = {}

        # Check ComfyUI connection
        if self.comfyui_client:
            status = self.comfyui_client.test_connection()
            services['comfyui'] = {
                'status': 'ok' if status['available'] else 'error',
                'message': f"Response time: {status['response_time']}s" if status['available'] else status['error']
            }
        else:
            services['comfyui'] = {
                'status': 'disabled',
                'message': 'ComfyUI client not initialized'
            }

        # Check output directory
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            services['output_directory'] = {
                'status': 'ok',
                'message': f'Writable: {self.output_dir}'
            }
        except Exception as e:
            services['output_directory'] = {
                'status': 'error',
                'message': str(e)
            }

        all_ok = all(s['status'] in ['ok', 'warning'] for s in services.values())

        return {
            'healthy': all_ok,
            'services': services
        }

    # ==================== Helper Methods ====================

    def _generate_simple_prompts(self, story_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate simple fallback prompts from story data."""
        title = story_data.get('title', 'Untitled Story')
        content = story_data.get('content', '')[:500]

        # Extract key themes
        themes = "cyberpunk interdimensional journalism"

        prompts = {
            'feature': f"Epic feature image for story: {title}. Themes: {themes}",
            'beginning': f"Opening scene from: {title}. Setting up the narrative.",
            'middle': f"Peak moment from: {title}. Height of action and conflict.",
            'end': f"Resolution of: {title}. Aftermath and new status quo."
        }

        return prompts

    # ==================== MCP Tool Handlers ====================

    async def handle_test_comfyui_connection(self, arguments: dict) -> list[TextContent]:
        """Test ComfyUI connection and return status."""
        self.log_tool_call("test_comfyui_connection", arguments)

        try:
            if not self.comfyui_client:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        'available': False,
                        'error': 'ComfyUI client not initialized'
                    }, indent=2)
                )]

            status = self.comfyui_client.test_connection()

            return [TextContent(
                type="text",
                text=json.dumps(status, indent=2)
            )]
        except Exception as e:
            error = self.handle_error(e, "test_comfyui_connection")
            return [TextContent(
                type="text",
                text=json.dumps(error, indent=2)
            )]

    async def handle_generate_single_image(self, arguments: dict) -> list[TextContent]:
        """Generate a single image from a prompt."""
        self.log_tool_call("generate_single_image", arguments)

        try:
            prompt = arguments.get('prompt')
            output_filename = arguments.get('output_filename')
            width = arguments.get('width', 1344)
            height = arguments.get('height', 768)

            if not prompt:
                raise ValueError("prompt is required")
            if not output_filename:
                raise ValueError("output_filename is required")

            if not self.comfyui_client or not self.comfyui_client.is_available():
                raise RuntimeError("ComfyUI service not available")

            # Generate output path
            output_path = self.output_dir / output_filename

            # Generate image
            result_path = self.comfyui_client.generate_image(
                prompt=prompt,
                output_path=output_path,
                width=width,
                height=height
            )

            if result_path:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        'success': True,
                        'image_path': str(result_path),
                        'width': width,
                        'height': height
                    }, indent=2)
                )]
            else:
                raise RuntimeError("Image generation failed")

        except Exception as e:
            error = self.handle_error(e, "generate_single_image")
            return [TextContent(
                type="text",
                text=json.dumps(error, indent=2)
            )]

    async def handle_generate_image_prompts(self, arguments: dict) -> list[TextContent]:
        """Generate Stable Diffusion prompts from story data."""
        self.log_tool_call("generate_image_prompts", arguments)

        try:
            story_data = arguments.get('story_data')

            if not story_data:
                raise ValueError("story_data is required")

            # Generate prompts (simple version - could be enhanced with Gemini)
            prompts = self._generate_simple_prompts(story_data)

            return [TextContent(
                type="text",
                text=json.dumps(prompts, indent=2)
            )]
        except Exception as e:
            error = self.handle_error(e, "generate_image_prompts")
            return [TextContent(
                type="text",
                text=json.dumps(error, indent=2)
            )]

    async def handle_generate_story_images(self, arguments: dict) -> list[TextContent]:
        """Generate all images for a story (feature + 3 article images)."""
        self.log_tool_call("generate_story_images", arguments)

        try:
            story_id = arguments.get('story_id')
            story_data = arguments.get('story_data')

            if not story_id:
                raise ValueError("story_id is required")
            if not story_data:
                raise ValueError("story_data is required")

            if not self.comfyui_client or not self.comfyui_client.is_available():
                raise RuntimeError("ComfyUI service not available")

            # Get or generate prompts
            if 'image_prompts' in story_data:
                # Use provided prompts
                prompts = story_data['image_prompts']
                logger.info(f"Using provided image prompts for story {story_id}")
            else:
                # Generate prompts
                prompts = self._generate_simple_prompts(story_data)
                logger.info(f"Generated fallback prompts for story {story_id}")

            # Generate images
            images = {}
            image_sections = {
                'feature': prompts.get('feature_image_prompt', prompts.get('feature')),
                'beginning': prompts.get('beginning_image_prompt', prompts.get('beginning')),
                'middle': prompts.get('middle_image_prompt', prompts.get('middle')),
                'end': prompts.get('end_image_prompt', prompts.get('end'))
            }

            for section, prompt in image_sections.items():
                if not prompt:
                    logger.warning(f"No prompt found for {section} section")
                    continue

                output_filename = f"story_{story_id}_{section}.png"
                output_path = self.output_dir / output_filename

                logger.info(f"Generating {section} image for story {story_id}")
                result_path = self.comfyui_client.generate_image(
                    prompt=prompt,
                    output_path=output_path,
                    width=1344,
                    height=768
                )

                if result_path:
                    images[section] = str(result_path)
                    logger.info(f"Successfully generated {section} image")

                    # Add delay between generations
                    if section != 'end':
                        time.sleep(5)
                else:
                    logger.error(f"Failed to generate {section} image")
                    images[section] = None

            successful_images = len([p for p in images.values() if p])

            return [TextContent(
                type="text",
                text=json.dumps({
                    'success': True,
                    'story_id': story_id,
                    'images': images,
                    'successful_count': successful_images,
                    'total_count': len(images)
                }, indent=2)
            )]

        except Exception as e:
            error = self.handle_error(e, "generate_story_images")
            return [TextContent(
                type="text",
                text=json.dumps(error, indent=2)
            )]

    # ==================== MCP Server Setup ====================

    def setup_handlers(self):
        """Set up MCP tool and resource handlers."""

        # Register tool list handler
        @self.app.list_tools()
        async def list_tools() -> list[Tool]:
            """List all available tools."""
            return [
                Tool(
                    name="test_comfyui_connection",
                    description="Test ComfyUI connection and get health status",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="generate_single_image",
                    description="Generate a single image from a Stable Diffusion prompt using ComfyUI",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Stable Diffusion prompt for image generation"
                            },
                            "output_filename": {
                                "type": "string",
                                "description": "Filename for the generated image (e.g., 'story_123_feature.png')"
                            },
                            "width": {
                                "type": "integer",
                                "description": "Image width in pixels (default: 1344)",
                                "default": 1344
                            },
                            "height": {
                                "type": "integer",
                                "description": "Image height in pixels (default: 768)",
                                "default": 768
                            }
                        },
                        "required": ["prompt", "output_filename"]
                    }
                ),
                Tool(
                    name="generate_image_prompts",
                    description="Generate Stable Diffusion prompts from story data for feature and article images",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "story_data": {
                                "type": "object",
                                "description": "Story data dictionary with title, content, etc.",
                                "properties": {
                                    "title": {"type": "string"},
                                    "content": {"type": "string"},
                                    "character_id": {"type": "string"}
                                }
                            }
                        },
                        "required": ["story_data"]
                    }
                ),
                Tool(
                    name="generate_story_images",
                    description="Generate all images for a story: 1 feature image + 3 article images (beginning, middle, end). Returns paths to generated 1344x768 PNG images.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "story_id": {
                                "type": "string",
                                "description": "Unique story identifier"
                            },
                            "story_data": {
                                "type": "object",
                                "description": "Story data with title, content, and optional image_prompts",
                                "properties": {
                                    "title": {"type": "string"},
                                    "content": {"type": "string"},
                                    "character_id": {"type": "string"},
                                    "image_prompts": {
                                        "type": "object",
                                        "description": "Optional pre-generated prompts",
                                        "properties": {
                                            "feature_image_prompt": {"type": "string"},
                                            "beginning_image_prompt": {"type": "string"},
                                            "middle_image_prompt": {"type": "string"},
                                            "end_image_prompt": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        },
                        "required": ["story_id", "story_data"]
                    }
                )
            ]

        # Register call tool handler
        @self.app.call_tool()
        async def call_tool(name: str, arguments: dict) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
            """Handle tool calls."""
            handlers = {
                "test_comfyui_connection": self.handle_test_comfyui_connection,
                "generate_single_image": self.handle_generate_single_image,
                "generate_image_prompts": self.handle_generate_image_prompts,
                "generate_story_images": self.handle_generate_story_images,
            }

            handler = handlers.get(name)
            if handler:
                return await handler(arguments or {})
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def run(self):
        """Run the MCP server."""
        # Initialize
        self.initialize()

        # Setup handlers
        self.setup_handlers()

        self.logger.info("Starting Ephergent Image Generator MCP server...")
        self.logger.info(f"ComfyUI URL: {self.comfyui_url}")
        self.logger.info(f"Output directory: {self.output_dir}")

        # Run with stdio transport
        async with stdio_server() as (read_stream, write_stream):
            await self.app.run(
                read_stream,
                write_stream,
                self.app.create_initialization_options()
            )


async def main():
    """Main entry point."""
    server = ImageGeneratorServer()
    await server.run()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
