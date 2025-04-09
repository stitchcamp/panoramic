import cv2
import numpy as np
import glob
import argparse
import os

def crop_black_borders(image, threshold_val=10, margin=0):
    """
    Crops the black borders evenly from a panoramic image to create a clean rectangular result.
    
    Args:
        image: The input image (stitched panorama) as a NumPy array.
        threshold_val: Pixels with intensity below this value are considered black.
        margin: Margin to leave around the content (can be negative to crop more aggressively).
        
    Returns:
        The cropped image as a NumPy array, or the original image if cropping fails.
    """
    if image is None:
        print("Error: Input image is None.")
        return None
    
    print(f"Original size: {image.shape[1]}x{image.shape[0]}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a binary mask of non-black pixels
    mask = (gray > threshold_val).astype(np.uint8) * 255
    
    # Find the bounding box of non-black pixels for each row and column
    height, width = image.shape[:2]
    
    # For each row, find the leftmost and rightmost non-black pixel
    left_border = width - 1
    right_border = 0
    top_border = height - 1
    bottom_border = 0
    
    # Process rows to find top and bottom borders
    row_has_content = np.any(mask > 0, axis=1)
    content_rows = np.where(row_has_content)[0]
    if len(content_rows) > 0:
        top_border = content_rows[0]
        bottom_border = content_rows[-1]
    
    # Process columns to find left and right borders
    col_has_content = np.any(mask > 0, axis=0)
    content_cols = np.where(col_has_content)[0]
    if len(content_cols) > 0:
        left_border = content_cols[0]
        right_border = content_cols[-1]
    
    print(f"Content borders: left={left_border}, right={right_border}, top={top_border}, bottom={bottom_border}")
    
    # Apply margin (can be negative to crop more)
    left_border = max(0, left_border - margin)
    top_border = max(0, top_border - margin)
    right_border = min(width - 1, right_border + margin)
    bottom_border = min(height - 1, bottom_border + margin)
    
    # Make the crop symmetrical if possible
    # Calculate the horizontal and vertical centers
    h_center = width // 2
    v_center = height // 2
    
    # Calculate distances from center to borders
    left_dist = h_center - left_border
    right_dist = right_border - h_center
    top_dist = v_center - top_border
    bottom_dist = bottom_border - v_center
    
    # Use the minimum distance on each side to ensure symmetry
    h_min_dist = min(left_dist, right_dist)
    v_min_dist = min(top_dist, bottom_dist)
    
    # Calculate new borders that are equidistant from center
    left_border = max(0, h_center - h_min_dist)
    right_border = min(width - 1, h_center + h_min_dist)
    top_border = max(0, v_center - v_min_dist)
    bottom_border = min(height - 1, v_center + v_min_dist)
    
    print(f"Symmetrical borders: left={left_border}, right={right_border}, top={top_border}, bottom={bottom_border}")
    
    # Crop the image
    cropped_image = image[top_border:bottom_border+1, left_border:right_border+1]
    print(f"Cropped size: {cropped_image.shape[1]}x{cropped_image.shape[0]}")
    
    # Sanity check
    if cropped_image.shape[0] < 10 or cropped_image.shape[1] < 10:
        print("Warning: Cropped image is too small. Returning original.")
        return image
    
    return cropped_image

def create_action_shot(images, background_idx=0, threshold=30, kernel_size=5):
    """
    Creates an action shot by combining multiple frames showing a moving subject.
    
    Args:
        images: List of input images (must be the same size).
        background_idx: Index of the image to use as the background (default: first image).
        threshold: Threshold for detecting movement between frames.
        kernel_size: Size of morphological operations kernel for cleaning up masks.
        
    Returns:
        The combined action shot image.
    """
    if len(images) < 2:
        print("Error: Need at least 2 images for an action shot.")
        return None
    
    # Check that all images have the same dimensions
    height, width = images[0].shape[:2]
    for i, img in enumerate(images):
        if img.shape[:2] != (height, width):
            print(f"Error: Image {i} has different dimensions. All images must be the same size.")
            return None
    
    # Use the specified image as the background
    background_idx = min(background_idx, len(images) - 1)  # Ensure valid index
    result = images[background_idx].copy()
    background = images[background_idx].copy()
    
    # Create a kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    print(f"Creating action shot with {len(images)} frames...")
    print(f"Using frame {background_idx} as background")
    
    # Process each image (except the background)
    for i, img in enumerate(images):
        if i == background_idx:
            continue
            
        # Convert images to grayscale for comparison
        gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference between current image and background
        diff = cv2.absdiff(gray_bg, gray_img)
        
        # Apply threshold to get binary mask of differences
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Clean up the mask with morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Create 3-channel mask for color images
        mask_3ch = cv2.merge([mask, mask, mask])
        
        # Extract moving parts from current frame
        moving_parts = cv2.bitwise_and(img, mask_3ch)
        
        # Create inverse mask for background
        mask_inv = cv2.bitwise_not(mask_3ch)
        
        # Extract static parts from result
        static_parts = cv2.bitwise_and(result, mask_inv)
        
        # Combine moving parts with static parts
        result = cv2.add(static_parts, moving_parts)
        
        print(f"  Processed frame {i}")
    
    return result

def main(args):
    """
    Main function to load images, perform stitching or action shot, crop, and save/display.
    """
    print("--- Starting Image Processing ---")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output filename: {args.output_name}")
    print(f"Image extension: {args.ext}")
    print(f"Mode: {'Action Shot' if args.action_shot else 'Panorama Stitching'}")
    if args.action_shot:
        print(f"Background frame index: {args.background_idx}")
        print(f"Movement threshold: {args.movement_threshold}")
    if args.crop:
        print(f"Crop borders: {args.crop}")
        print(f"Crop threshold: {args.crop_thresh}")
    print(f"Display result: {args.display}")
    print("-" * 30)
 
    print("Loading images...")
    # Construct the search pattern carefully
    search_pattern = os.path.join(args.input_dir, f"*.{args.ext}")
    print(f"Searching for images with pattern: {search_pattern}")
    image_paths = sorted(glob.glob(search_pattern)) # Sort for consistent order
 
    if len(image_paths) < 2:
        print(f"Error: Need at least 2 images, found {len(image_paths)} in '{args.input_dir}' with extension '{args.ext}'")
        return
 
    images = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue
        images.append(img)
        print(f"  Loaded: {os.path.basename(image_path)} ({img.shape[1]}x{img.shape[0]})")
 
    if len(images) < 2:
        print(f"\nError: Need at least 2 valid images, loaded {len(images)}")
        return
    
    # Process based on selected mode
    if args.action_shot:
        print(f"\nCreating action shot from {len(images)} images...")
        output_image = create_action_shot(
            images, 
            background_idx=args.background_idx,
            threshold=args.movement_threshold,
            kernel_size=args.kernel_size
        )
        
        if output_image is None:
            print("Error: Action shot creation failed.")
            return
            
        print("Action shot created successfully!")
    else:
        print(f"\nAttempting to stitch {len(images)} images...")
        # Create a Stitcher object
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        
        # Perform stitching
        status, panorama = stitcher.stitch(images)
        
        # Check stitching status
        if status == cv2.Stitcher_OK:
            print("\nStitching successful!")
            output_image = panorama
        else:
            handle_stitching_error(status)
            return
    
    # Crop black borders if requested
    if args.crop:
        print("\nCropping black borders...")
        cropped_image = crop_black_borders(
            output_image, threshold_val=args.crop_thresh, margin=args.margin
        )
        # Check if cropping returned a valid image
        if cropped_image is not None and cropped_image.size > 0:
             print(f"Final cropped size: {cropped_image.shape[1]}x{cropped_image.shape[0]}")
             output_image = cropped_image # Use the cropped version
        else:
             print("Warning: Cropping failed or resulted in empty image, using original image.")
             print(f"Final size (uncropped): {output_image.shape[1]}x{output_image.shape[0]}")
    else:
        print(f"\nFinal size (uncropped): {output_image.shape[1]}x{output_image.shape[0]}")
    
    # Save the result
    output_path = os.path.join(args.output_dir, args.output_name)
    try:
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        cv2.imwrite(output_path, output_image)
        print(f"\nImage saved successfully to: {output_path}")
    except Exception as e:
        print(f"\nError: Failed to save image to {output_path}")
        print(f"Reason: {e}")
        return # Exit if fails
    
    # Optionally display the result
    if args.display:
        print("\nDisplaying result (press any key to close)...")
        try:
            # Resize for display if it's too large to fit on screen reasonably
            h, w = output_image.shape[:2]
            max_display_width = 1200
            max_display_height = 800
            scale = 1.0
            if w > max_display_width:
                scale = max_display_width / w
            if h * scale > max_display_height:
                scale = max_display_height / h
            
            if scale < 1.0:
                display_h = int(h * scale)
                display_w = int(w * scale)
                print(f"  Resizing for display to {display_w}x{display_h}")
                display_img = cv2.resize(output_image, (display_w, display_h), interpolation=cv2.INTER_AREA)
            else:
                display_img = output_image
            
            cv2.imshow("Result", display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"\nError: Failed to display image.")
            print(f"Reason: {e}")
    
    print("\n--- Processing finished ---")

def handle_stitching_error(status):
    """Handle and print informative messages for stitching errors"""
    if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        print("\nStitching failed: Not enough keypoints were detected or matched.")
        print("Tips: Try using images with more distinctive features, better overlap, or consistent lighting.")
    elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        print("\nStitching failed: Homography estimation failed.")
        print("Tips: Ensure images have sufficient overlap (30-50% recommended) and were taken from roughly the same viewpoint (rotation, not translation).")
    elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
         print("\nStitching failed: Camera parameter adjustment failed.")
         print("Tips: This can sometimes happen with complex scenes or lens distortions.")
    else:
        print(f"\nStitching failed with status code: {status}")
        print("Tips: Check image quality, overlap, order, and consistency.")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Process multiple images into a panorama or action shot using OpenCV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
 
    # Input/Output Arguments
    parser.add_argument(
        "-i", "--input_dir",
        required=True,
        help="Directory containing the images to process."
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="output", # Default output directory name
        help="Directory to save the resulting image."
    )
    parser.add_argument(
        "-n", "--output_name",
        default="result.jpg", # Default output filename
        help="Filename for the output image."
    )
    parser.add_argument(
        "-e", "--ext",
        default="jpg", # Default image extension
        help="Extension of the input image files (e.g., jpg, png, tif)."
    )
 
    # Mode Selection
    parser.add_argument(
        "--action_shot",
        action="store_true",
        help="Create an action shot instead of a panorama."
    )
    
    # Action Shot Parameters
    parser.add_argument(
        "--background_idx",
        type=int,
        default=0,
        help="Index of the image to use as background for action shot (0 = first image)."
    )
    parser.add_argument(
        "--movement_threshold",
        type=int,
        default=30,
        help="Threshold for detecting movement in action shot (0-255)."
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=5,
        help="Size of kernel for morphological operations in action shot."
    )
    
    # Cropping Arguments
    parser.add_argument(
        "--crop",
        action="store_true", # Makes it a flag: presence means True
        help="Automatically crop black borders from the result."
    )
    parser.add_argument(
        "--crop_thresh",
        type=int,
        default=5, # Default threshold value for cropping
        help="Intensity threshold (0-255) for cropping black borders. "
             "Pixels below this value are considered black. Adjust if cropping is inaccurate."
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=5, # Default margin value for cropping
        help="Margin to leave around the content when cropping. "
             "Can be negative to crop more aggressively."
    )
 
    # Display Argument
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the resulting image in a window after processing."
    )
 
    # Parse arguments from cmd
    args = parser.parse_args()
 
    # Run the main function
    main(args)
