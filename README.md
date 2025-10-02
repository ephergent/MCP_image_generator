# MCP Image Generator Server

Generate images using ComfyUI for The Ephergent story generation workflow.

## Overview

This MCP server provides image generation services using ComfyUI, a powerful node-based image generation system. It generates high-quality images for Ephergent stories including:

- 1 feature image (1344x768)
- 3 article images: beginning, middle, end (1344x768 each)

## Features

- ComfyUI integration with WebSocket progress tracking
- Automatic style prefix/suffix application (Ephergent universe styling)
- Health check and connection testing
- Configurable image dimensions
- Support for custom Stable Diffusion prompts

## MCP Tools

### 1. test_comfyui_connection
Test ComfyUI connection and get health status.

**Parameters:** None

**Returns:** Connection status with response time and availability

### 2. generate_single_image
Generate a single image from a Stable Diffusion prompt.

**Parameters:**
- `prompt` (required): Stable Diffusion prompt
- `output_filename` (required): Filename for generated image
- `width` (optional): Image width in pixels (default: 1344)
- `height` (optional): Image height in pixels (default: 768)

**Returns:** Image path and generation metadata

### 3. generate_image_prompts
Generate Stable Diffusion prompts from story data.

**Parameters:**
- `story_data` (required): Story data with title, content, character_id

**Returns:** Map of image types to prompts (feature, beginning, middle, end)

### 4. generate_story_images
Generate all images for a story (1 feature + 3 article images).

**Parameters:**
- `story_id` (required): Unique story identifier
- `story_data` (required): Story data including:
  - `title`: Story title
  - `content`: Story content
  - `character_id`: Optional character ID
  - `image_prompts`: Optional pre-generated prompts

**Returns:** Paths to all generated images and success count

## Installation

### Prerequisites

1. ComfyUI instance running locally or remotely
2. Python 3.11+
3. uv (recommended) or pip

### Setup

1. Ensure ComfyUI is running and accessible
2. Configure environment variables in `.env`:

```bash
COMFYUI_URL=http://comfyui.nexus.home.test
IMAGE_OUTPUT_DIR=/tmp/ephergent_images
```

3. Install dependencies (handled automatically by uv):

```bash
uv run server.py
```

### Claude Desktop Configuration

Add to your Claude Desktop config file:

```json
{
  "mcpServers": {
    "ephergent-image-generator": {
      "command": "uv",
      "args": ["run", "/absolute/path/to/MCP_image_generator/server.py"],
      "env": {
        "COMFYUI_URL": "http://comfyui.nexus.home.test",
        "IMAGE_OUTPUT_DIR": "/tmp/ephergent_images"
      }
    }
  }
}
```

## Usage

### Testing Connection

```python
# Test if ComfyUI is available
result = await mcp.call_tool("test_comfyui_connection", {})
```

### Generating Story Images

```python
# Generate all images for a story
result = await mcp.call_tool("generate_story_images", {
    "story_id": "story_123",
    "story_data": {
        "title": "The Quantum Coffee Crisis",
        "content": "In a dimension where coffee beans...",
        "character_id": "pixel_paradox",
        "image_prompts": {
            "feature_image_prompt": "Epic scene with...",
            "beginning_image_prompt": "Opening scene...",
            "middle_image_prompt": "Peak action...",
            "end_image_prompt": "Resolution..."
        }
    }
})
```

### Generating Single Image

```python
# Generate a single custom image
result = await mcp.call_tool("generate_single_image", {
    "prompt": "A cyberpunk newsroom with interdimensional portals",
    "output_filename": "custom_image.png",
    "width": 1344,
    "height": 768
})
```

## Technical Details

### ComfyUI Integration

- Uses workflow template: `assets/workflows/t2i_ephergent_season_03_workflow.json`
- Applies Ephergent style prefix: "ArsMJStyle, dnddarkestfantasy, Kenva, fluxlisimo..."
- Applies post-style suffix with anime manga styling instructions
- Random seed generation for variation
- WebSocket-based progress tracking

### Image Specifications

- Default resolution: 1344x768 (wide format)
- Format: PNG
- Color space: RGB
- Style: Stylized 3D anime manga with painterly cel-shading

### Performance

- Average generation time: ~200 seconds per image
- Total story generation: ~15 minutes (4 images)
- Timeout: 5 minutes per image
- 5-second delay between consecutive generations

## Environment Variables

- `COMFYUI_URL`: ComfyUI server URL (required)
- `COMFYUI_ENABLED`: Enable/disable ComfyUI (default: true)
- `IMAGE_OUTPUT_DIR`: Output directory for images (default: /tmp/ephergent_images)

## Dependencies

Automatically managed via PEP 723 inline dependencies:

- mcp>=0.9.0
- python-dotenv>=1.0.0
- requests>=2.31.0
- websocket-client>=1.6.0
- Pillow>=10.0.0

## Error Handling

The server includes comprehensive error handling for:

- ComfyUI connection failures
- Image generation timeouts
- WebSocket disconnections
- Invalid prompts or parameters
- Disk space issues

## Reference Implementation

Based on:
- `/reference_code/comfyui_service.py`
- `/reference_code/image_service.py`

## License

See LICENSE file in repository root
