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
Test script for MCP Image Generator Server

Tests basic functionality without requiring MCP client.
"""

import sys
import os
import json
import asyncio
from pathlib import Path

# Set up environment
os.environ['COMFYUI_URL'] = 'http://comfyui.nexus.home.test'
os.environ['IMAGE_OUTPUT_DIR'] = '/tmp/ephergent_test_images'

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from server import ImageGeneratorServer


async def test_server():
    """Test server initialization and configuration."""
    print("=" * 60)
    print("MCP Image Generator Server - Test Suite")
    print("=" * 60)

    # Initialize server
    print("\n1. Initializing server...")
    server = ImageGeneratorServer()
    print(f"   ✓ Server initialized")
    print(f"   - ComfyUI URL: {server.comfyui_url}")
    print(f"   - Output directory: {server.output_dir}")
    print(f"   - ComfyUI enabled: {server.comfyui_enabled}")

    # Validate configuration
    print("\n2. Validating configuration...")
    config = server.validate_configuration()
    print(f"   - Valid: {config['valid']}")
    if config['errors']:
        print(f"   - Errors: {config['errors']}")
    if config['warnings']:
        print(f"   - Warnings: {config['warnings']}")

    # Check health
    print("\n3. Checking health...")
    health = await server.check_health()
    print(f"   - Healthy: {health['healthy']}")
    for service_name, service_info in health['services'].items():
        status_icon = "✓" if service_info['status'] == 'ok' else "✗"
        print(f"   {status_icon} {service_name}: {service_info['message']}")

    # Test connection
    print("\n4. Testing ComfyUI connection...")
    try:
        result = await server.handle_test_comfyui_connection({})
        data = json.loads(result[0].text)
        print(f"   - Available: {data.get('available', False)}")
        print(f"   - Response time: {data.get('response_time', 'N/A')}")
        if data.get('error'):
            print(f"   - Error: {data['error']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test prompt generation
    print("\n5. Testing prompt generation...")
    try:
        test_story = {
            'title': 'The Quantum Coffee Crisis',
            'content': 'In a dimension where coffee beans are sentient...',
            'character_id': 'pixel_paradox'
        }
        result = await server.handle_generate_image_prompts({'story_data': test_story})
        prompts = json.loads(result[0].text)
        print(f"   ✓ Generated {len(prompts)} prompts")
        for key in prompts.keys():
            print(f"     - {key}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n" + "=" * 60)
    print("Test suite complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_server())
