# MCP Image Generator - Quick Start Guide

## Installation (5 minutes)

### 1. Verify Prerequisites
```bash
# Check Python version (need 3.11+)
python3 --version

# Check ComfyUI is running
curl http://comfyui.nexus.home.test/system_stats
```

### 2. Configure Environment
Add to your `.env` file (in the repo root):
```bash
COMFYUI_URL=http://comfyui.nexus.home.test
IMAGE_OUTPUT_DIR=/tmp/ephergent_images
```

### 3. Test the Server
```bash
# From MCP_servers directory
python3 MCP_image_generator/test_server.py
```

Expected output:
```
✓ Server initialized
✓ Configuration validation: Valid
✓ Health check: Healthy
✓ ComfyUI connection: Available
✓ Prompt generation: 4 prompts generated
```

## Claude Desktop Setup

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ephergent-image-generator": {
      "command": "uv",
      "args": [
        "run",
        "/Users/jeremy/Documents/ephergent_next/MCP_servers/MCP_image_generator/server.py"
      ],
      "env": {
        "COMFYUI_URL": "http://comfyui.nexus.home.test",
        "IMAGE_OUTPUT_DIR": "/tmp/ephergent_images"
      }
    }
  }
}
```

Restart Claude Desktop.

## Basic Usage

### Test Connection
```python
# In Claude Desktop, ask:
"Test the ComfyUI connection using the image generator"

# Claude will call:
test_comfyui_connection()
```

### Generate Story Images
```python
# Ask Claude:
"Generate images for this story:
Title: The Quantum Coffee Crisis
Content: [story text]
Character: pixel_paradox"

# Claude will call:
generate_story_images({
    "story_id": "story_001",
    "story_data": {
        "title": "The Quantum Coffee Crisis",
        "content": "...",
        "character_id": "pixel_paradox"
    }
})
```

### Generate Single Image
```python
# Ask Claude:
"Generate an image of a cyberpunk newsroom"

# Claude will call:
generate_single_image({
    "prompt": "A cyberpunk newsroom with interdimensional portals",
    "output_filename": "newsroom.png",
    "width": 1344,
    "height": 768
})
```

## Workflow

### Complete Story Generation
1. **Get story from MCP_story_generator**
2. **Generate image prompts** (optional, can use story content directly)
3. **Generate all 4 images** (feature + beginning + middle + end)
4. **Images saved to** `IMAGE_OUTPUT_DIR`
5. **Pass image paths to** MCP_video_generator

### Expected Timing
- Feature image: ~3-5 minutes
- Beginning image: ~3-5 minutes
- Middle image: ~3-5 minutes
- End image: ~3-5 minutes
- **Total: 12-20 minutes for 4 images**

## Troubleshooting

### ComfyUI Not Available
```bash
# Check if ComfyUI is running
curl http://comfyui.nexus.home.test/system_stats

# If not running, start ComfyUI
# (instructions depend on your ComfyUI setup)
```

### Connection Timeout
```bash
# Increase timeout in server.py:
# Find: timeout_duration = 300
# Change to: timeout_duration = 600
```

### Output Directory Permission Error
```bash
# Create directory with proper permissions
mkdir -p /tmp/ephergent_images
chmod 755 /tmp/ephergent_images
```

### Workflow Template Not Found
```bash
# Verify workflow file exists
ls -l assets/workflows/t2i_ephergent_season_03_workflow.json

# If missing, check reference_code for backup
```

## Image Specifications

### Default Settings
- **Resolution:** 1344 x 768 (wide format)
- **Format:** PNG
- **Color Space:** RGB
- **File Size:** ~2-5 MB per image

### Style Characteristics
- Stylized 3D anime manga
- Painterly cel-shading
- Volumetric lighting
- Soft watercolor gradients
- Comic book halftone patterns
- Cinematic rim lighting

### Customization
To change default dimensions, modify the tool call:
```python
generate_single_image({
    "prompt": "...",
    "output_filename": "custom.png",
    "width": 1920,   # Custom width
    "height": 1080   # Custom height
})
```

## Advanced Usage

### Pre-Generated Prompts
If you have prompts from another service:
```python
generate_story_images({
    "story_id": "story_001",
    "story_data": {
        "title": "...",
        "content": "...",
        "image_prompts": {
            "feature_image_prompt": "Epic cyberpunk scene...",
            "beginning_image_prompt": "Opening establishing shot...",
            "middle_image_prompt": "Action peak moment...",
            "end_image_prompt": "Resolution aftermath..."
        }
    }
})
```

### Batch Processing
For multiple stories, call `generate_story_images` sequentially:
```python
for story in stories:
    generate_story_images({
        "story_id": story['id'],
        "story_data": story
    })
    # 15-minute wait between stories
```

## Files and Locations

### Generated Images
Located in `IMAGE_OUTPUT_DIR` with naming pattern:
```
story_{story_id}_feature.png
story_{story_id}_beginning.png
story_{story_id}_middle.png
story_{story_id}_end.png
```

### Logs
Server logs go to stderr, captured by MCP client:
```
INFO:image_generator:Starting image generation...
INFO:image_generator:Generation progress: 45.2%
INFO:image_generator:Image successfully saved to: /tmp/ephergent_images/story_001_feature.png
```

## Performance Tips

### Optimize Generation Time
1. Ensure ComfyUI has adequate GPU memory
2. Close other GPU-intensive applications
3. Use local ComfyUI instance when possible (faster than remote)

### Disk Space Management
```bash
# Check available space
df -h /tmp/ephergent_images

# Clean up old images
find /tmp/ephergent_images -mtime +7 -delete
```

### Concurrent Requests
The server processes one image at a time. For parallel generation:
- Run multiple server instances on different ports
- Use different `IMAGE_OUTPUT_DIR` for each instance

## Integration with Other Servers

### With MCP_story_generator
```python
# 1. Generate story
story = call_tool("generate_story", {...})

# 2. Generate images
images = call_tool("generate_story_images", {
    "story_id": story['id'],
    "story_data": story
})
```

### With MCP_video_generator
```python
# 1. Generate images
images = call_tool("generate_story_images", {...})

# 2. Create video from images
video = call_tool("create_story_video", {
    "story_id": story['id'],
    "images": images['images']
})
```

## Next Steps

1. ✅ Test connection
2. ✅ Generate first story images
3. ✅ Verify output quality
4. ✅ Integrate with story generator
5. ✅ Connect to video generator
6. ✅ Build complete pipeline

## Support

For issues or questions:
- Check logs in Claude Desktop developer tools
- Review IMPLEMENTATION_REPORT.md for technical details
- See README.md for comprehensive documentation
- Examine reference_code/comfyui_service.py for implementation details
