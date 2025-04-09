### Important Notes for Panorama Stitching:
**The images should have sufficient overlap (30-50% is recommended)**
**Images should be taken from approximately the same position (rotating the camera, not moving it)**
**Consistent lighting across images helps with better stitching**
**The script automatically sorts the images by filename, so naming your files in sequence (e.g., img001.jpg, img002.jpg) ensures proper ordering**

**Command for panorama:**
```powershell
python IT322_BSIT3A_Lab3_CatalanJ_BonsoJL.py -i images --crop --crop_thresh 225 --margin -70
```
**Adjust 'images' for image folder location**
**Adjust '--crop_thresh' and '--margin' values to remove black edges**


# Important Notes for Action Shot

When creating action shots with this script, keep these key points in mind for best results:

1. **Camera Stability**
   - Use a tripod or very stable surface
   - Any camera movement between shots will create artifacts in the final image
   - Remote trigger or timer can help avoid camera shake

2. **Background**
   - The background should be static across all frames
   - Choose a clean, uncluttered background when possible
   - Avoid scenes with moving elements (like trees in wind, water, etc.)

3. **Subject Movement**
   - The subject should move across the frame in a clear path
   - Ensure good separation between different positions of the subject
   - Movement should be significant enough to detect but not so fast it blurs

4. **Lighting**
   - Consistent lighting across all frames is crucial
   - Avoid auto-exposure if possible (use manual settings)
   - Shadows can change between frames and cause detection issues

5. **Timing**
   - Take photos at regular intervals for smooth action progression
   - Faster movements need more frequent captures
   - Consider the full path you want to show and space shots accordingly

6. **Parameter Tuning**
   - `--movement_threshold`: Lower values (10-20) detect subtle movements but may include noise; higher values (40-50) require more significant movement
   - `--background_idx`: Choose the frame where the subject is either absent or least prominent
   - `--kernel_size`: Larger values (7-9) create smoother masks but less detail; smaller values (3-5) preserve detail but may be noisier

7. **Image Sequence**
   - The script processes images in filename order
   - Name your files sequentially (e.g., shot001.jpg, shot002.jpg)
   - Consider which frame to use as background when planning your sequence

8. **Technical Considerations**
   - All images must have identical dimensions
   - Consistent camera settings (focus, aperture, shutter speed) produce better results
   - RAW images converted to the same format work better than JPEGs with varying compression

9. **Subject Considerations**
   - Subjects with high contrast against the background work best
   - Avoid clothing/subjects that match the background color
   - More distinct movements create more dramatic action shots

10. **Post-Processing**
    - The script output may benefit from additional editing
    - You might need to manually clean up areas where detection wasn't perfect
    - Consider adjusting contrast/brightness of the final image

For best results, start with a test sequence in controlled conditions to understand how the parameters affect your specific shooting situation.

**Command for action image:**
```powershell
python IT322_BSIT3A_Lab3_CatalanJ_BonsoJL.py -i input_folder -o output_folder --action_shot --background_idx 0 --movement_threshold 30 --display

```