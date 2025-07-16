import glob
import os
import shutil

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def extract_clothing_region(image_path, region_ratio=0.6):
    """Extract clothing region from center of image"""
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    else:
        image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)

    height, width = image.shape[:2]

    # Extract center region (avoid background and extremities)
    margin = int(min(height, width) * (1 - region_ratio) / 2)
    center_region = image[margin : height - margin, margin : width - margin]

    return center_region


def extract_dominant_colors(image_path, n_colors=3):
    """Extract dominant colors using K-means clustering"""
    clothing_region = extract_clothing_region(image_path)

    # Resize for speed
    clothing_region = cv2.resize(clothing_region, (64, 64))

    # Reshape to pixel array
    pixels = clothing_region.reshape(-1, 3)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Convert BGR to RGB
    dominant_colors = [(int(c[2]), int(c[1]), int(c[0])) for c in kmeans.cluster_centers_]

    return dominant_colors


def calculate_color_similarity(color1, color2):
    """Calculate color similarity using Euclidean distance"""
    distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))
    # Normalize to 0-1 range (max distance is sqrt(3*255^2) = 441.67)
    similarity = max(0, 1 - (distance / 441.67))
    return similarity


class ColorBasedStaffDetector:
    """Color-based staff detector using dominant color matching"""

    def __init__(self, n_colors=3):
        """
        Initialize detector
        Args:
            n_colors: Number of dominant colors to extract
        """
        self.n_colors = n_colors
        self.uniform_colors = None
        print(f"Color-based Staff Detector initialized")
        print(f"Number of colors to extract: {n_colors}")

    def extract_features(self, image_path):
        """Extract dominant colors as features"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        colors = extract_dominant_colors(image_path, self.n_colors)
        return colors

    def calculate_similarity(self, uniform_path, person_path):
        """Calculate color similarity between uniform and person images"""
        uniform_colors = self.extract_features(uniform_path)
        person_colors = self.extract_features(person_path)

        # Store uniform colors for reference
        self.uniform_colors = uniform_colors

        # Calculate maximum similarity across all color combinations
        max_similarity = 0

        for uniform_color in uniform_colors:
            for person_color in person_colors:
                similarity = calculate_color_similarity(uniform_color, person_color)
                max_similarity = max(max_similarity, similarity)

        return max_similarity

    def process_directory(self, uniform_path, input_dir, output_dir, threshold=0.6):
        """
        Process multiple images in directory and classify as staff/non-staff

        Args:
            uniform_path: Path to uniform reference image
            input_dir: Directory containing images to process
            output_dir: Output directory for classification results
            threshold: Similarity threshold for staff detection

        Returns:
            dict: Processing results with statistics
        """
        # Supported image extensions
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]

        # Get all image files from input directory
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

        if not image_files:
            raise ValueError(f"No image files found in: {input_dir}")

        print(f"Found {len(image_files)} images to process")

        # Create output directories
        staff_dir = os.path.join(output_dir, "staff")
        non_staff_dir = os.path.join(output_dir, "non_staff")
        os.makedirs(staff_dir, exist_ok=True)
        os.makedirs(non_staff_dir, exist_ok=True)

        # Extract uniform colors once
        print("Extracting uniform colors...")
        uniform_colors = self.extract_features(uniform_path)
        self.uniform_colors = uniform_colors

        print(f"Uniform colors (RGB): {uniform_colors}")

        # Process each image
        results = []
        staff_count = 0

        print("Processing images...")
        for i, image_path in enumerate(image_files):
            try:
                # Extract colors and calculate similarity
                person_colors = self.extract_features(image_path)

                # Calculate maximum similarity
                max_similarity = 0
                best_match = None

                for uniform_color in uniform_colors:
                    for person_color in person_colors:
                        similarity = calculate_color_similarity(uniform_color, person_color)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_match = (uniform_color, person_color)

                # Determine classification
                is_staff = max_similarity > threshold

                # Copy file to appropriate directory
                filename = os.path.basename(image_path)
                if is_staff:
                    dest_path = os.path.join(staff_dir, filename)
                    staff_count += 1
                else:
                    dest_path = os.path.join(non_staff_dir, filename)

                shutil.copy2(image_path, dest_path)

                # Store result
                results.append(
                    {
                        "filename": filename,
                        "similarity": max_similarity,
                        "is_staff": is_staff,
                        "source_path": image_path,
                        "dest_path": dest_path,
                        "person_colors": person_colors,
                        "best_match": best_match,
                    }
                )

                # Progress indicator
                print(
                    f"  [{i+1:3d}/{len(image_files)}] {filename}: {max_similarity:.4f} -> {'STAFF' if is_staff else 'NON-STAFF'}"
                )

            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                continue

        # Calculate statistics
        total_processed = len(results)
        non_staff_count = total_processed - staff_count
        staff_detection_rate = (staff_count / total_processed) * 100 if total_processed > 0 else 0

        # Summary
        summary = {
            "total_images": len(image_files),
            "processed_images": total_processed,
            "staff_count": staff_count,
            "non_staff_count": non_staff_count,
            "staff_detection_rate": staff_detection_rate,
            "threshold": threshold,
            "uniform_colors": uniform_colors,
            "results": results,
        }

        return summary


def print_summary(summary):
    """Print processing summary"""
    print("\n" + "=" * 60)
    print("COLOR-BASED PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total images found:     {summary['total_images']}")
    print(f"Successfully processed: {summary['processed_images']}")
    print(f"Staff detected:         {summary['staff_count']}")
    print(f"Non-staff detected:     {summary['non_staff_count']}")
    print(f"Staff detection rate:   {summary['staff_detection_rate']:.2f}%")
    print(f"Threshold used:         {summary['threshold']}")
    print(f"Uniform colors (RGB):   {summary['uniform_colors']}")
    print("=" * 60)


def print_detailed_results(summary, show_top_n=5):
    """Print detailed results for top matches"""
    results = summary["results"]

    # Sort by similarity (descending)
    staff_results = [r for r in results if r["is_staff"]]
    non_staff_results = [r for r in results if not r["is_staff"]]

    staff_results.sort(key=lambda x: x["similarity"], reverse=True)
    non_staff_results.sort(key=lambda x: x["similarity"], reverse=True)

    print(f"\n📊 DETAILED RESULTS")
    print("-" * 50)

    if staff_results:
        print(f"🟢 Top {min(show_top_n, len(staff_results))} STAFF matches:")
        for i, result in enumerate(staff_results[:show_top_n]):
            print(f"  {i+1}. {result['filename']}: {result['similarity']:.4f}")
            if result["best_match"]:
                uniform_color, person_color = result["best_match"]
                print(f"     Uniform: RGB{uniform_color} ↔ Person: RGB{person_color}")

    if non_staff_results:
        print(f"\n🔴 Top {min(show_top_n, len(non_staff_results))} NON-STAFF matches:")
        for i, result in enumerate(non_staff_results[:show_top_n]):
            print(f"  {i+1}. {result['filename']}: {result['similarity']:.4f}")
            if result["best_match"]:
                uniform_color, person_color = result["best_match"]
                print(f"     Uniform: RGB{uniform_color} ↔ Person: RGB{person_color}")


def main():
    """Main function"""
    # ===== CONFIGURATION =====
    UNIFORM_IMAGE_PATH = "test/uniform_images/uniform5.jpg"  # Path to uniform reference image

    # For single image processing
    PERSON_IMAGE_PATH = "staff1.jpg"  # Path to person image to check

    # For batch processing
    INPUT_DIRECTORY = "test/person_images"  # Directory containing images to proces
    OUTPUT_DIRECTORY = "results_color"  # Output directory for classification

    # Color detection parameters
    N_COLORS = 3  # Number of dominant colors to extract
    THRESHOLD = 0.6  # Color similarity threshold for staff detection

    # Processing mode
    BATCH_MODE = True  # Set to True for batch processing, False for single image

    try:
        # Initialize detector
        detector = ColorBasedStaffDetector(n_colors=N_COLORS)

        if BATCH_MODE:
            # Batch processing mode
            print(f"\n🔄 BATCH PROCESSING MODE (Color-Based)")
            print(f"Uniform image: {UNIFORM_IMAGE_PATH}")
            print(f"Input directory: {INPUT_DIRECTORY}")
            print(f"Output directory: {OUTPUT_DIRECTORY}")
            print(f"Color threshold: {THRESHOLD}")
            print(f"Number of colors: {N_COLORS}")

            # Process directory
            summary = detector.process_directory(
                uniform_path=UNIFORM_IMAGE_PATH,
                input_dir=INPUT_DIRECTORY,
                output_dir=OUTPUT_DIRECTORY,
                threshold=THRESHOLD,
            )

            # Print summaries
            print_summary(summary)
            print_detailed_results(summary, show_top_n=3)

            # Show file distribution
            print(f"\n📁 Files have been sorted into:")
            print(f"  🟢 Staff:     {os.path.join(OUTPUT_DIRECTORY, 'staff')}")
            print(f"  🔴 Non-staff: {os.path.join(OUTPUT_DIRECTORY, 'non_staff')}")

        else:
            # Single image processing mode
            print(f"\n🔄 SINGLE IMAGE MODE (Color-Based)")
            print("Calculating color similarity...")

            similarity = detector.calculate_similarity(UNIFORM_IMAGE_PATH, PERSON_IMAGE_PATH)

            # Results
            is_staff = similarity > THRESHOLD
            status = "✓ STAFF" if is_staff else "✗ NOT STAFF"

            print(f"\nResults:")
            print(f"Uniform image: {UNIFORM_IMAGE_PATH}")
            print(f"Person image:  {PERSON_IMAGE_PATH}")
            print(f"Color similarity: {similarity:.6f}")
            print(f"Threshold:     {THRESHOLD}")
            print(f"Decision:      {status}")

            # Show extracted colors
            if detector.uniform_colors:
                print(f"Uniform colors (RGB): {detector.uniform_colors}")

            person_colors = detector.extract_features(PERSON_IMAGE_PATH)
            print(f"Person colors (RGB):  {person_colors}")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure the following files/directories exist:")
        print(f"   - Uniform image: {UNIFORM_IMAGE_PATH}")
        if BATCH_MODE:
            print(f"   - Input directory: {INPUT_DIRECTORY}")
        else:
            print(f"   - Person image: {PERSON_IMAGE_PATH}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
