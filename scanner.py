import cv2
import numpy as np
import time
import os
from datetime import datetime

class DocumentScanner:
    """
    A class to perform real-time document scanning and quality checks.

    This class uses a robust, contour-based pipeline to find the largest
    4-sided, card-shaped object in the frame. This logic is fast and
    does not require any templates or models.

    Feedback includes:
    - "Place document in frame"
    - "Move document closer"
    - "Move document farther"
    - "Center document"
    - "Too blurry"
    - "Avoid reflections"
    - "Hold still..."
    """

    def __init__(self,
                 blur_thresh=40.0,
                 glare_thresh_val=240,
                 glare_thresh_percent=0.01,
                 min_area_ratio=0.15,
                 max_area_ratio=0.8,
                 center_thresh_ratio=0.1,
                 capture_frames_thresh=30,
                 target_aspect_ratio=1.586,
                 aspect_ratio_tolerance=0.15,
                 gaussian_blur_kernel=(7, 7),
                 canny_thresh_1=50,
                 canny_thresh_2=150,
                 dilate_iterations=2,
                 erode_iterations=1,
                 approx_poly_epsilon_ratio=0.02):
        """
        Initializes the scanner with configurable thresholds.

        Args:
            blur_thresh (float): Laplacian variance threshold. Lower is blurrier.
            glare_thresh_val (int): Pixel intensity to be considered glare (0-255).
            glare_thresh_percent (float): Pct of doc area allowed to be glare.
            min_area_ratio (float): Min doc area / frame area.
            max_area_ratio (float): Max doc area / frame area.
            center_thresh_ratio (float): Allowed distance from center.
            capture_frames_thresh (int): Frames to "Hold still" before capture.

            -- Contour Detection Tuning --
            target_aspect_ratio (float): The target w/h ratio of the document.
            aspect_ratio_tolerance (float): Allowed % difference from the target.
            gaussian_blur_kernel (tuple): Kernel size for Gaussian blur.
            canny_thresh_1 (int): First threshold for Canny edge detector.
            canny_thresh_2 (int): Second threshold for Canny edge detector.
            dilate_iterations (int): Number of dilations to close edge gaps.
            erode_iterations (int): Number of erosions to refine edges.
            approx_poly_epsilon_ratio (float): Epsilon ratio for polygon approximation.
        """
        self.blur_thresh = blur_thresh
        self.glare_thresh_val = glare_thresh_val
        self.glare_thresh_percent = glare_thresh_percent
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.center_thresh_ratio = center_thresh_ratio
        self.capture_frames_thresh = capture_frames_thresh

        self.target_aspect_ratio = target_aspect_ratio
        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        self.gaussian_blur_kernel = gaussian_blur_kernel
        self.canny_thresh_1 = canny_thresh_1
        self.canny_thresh_2 = canny_thresh_2
        self.dilate_iterations = dilate_iterations
        self.erode_iterations = erode_iterations
        self.approx_poly_epsilon_ratio = approx_poly_epsilon_ratio

        # --- Initialize Webcam ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_area = self.frame_h * self.frame_w
        self.frame_center = (self.frame_w // 2, self.frame_h // 2)
        self.center_thresh_pixels = self.center_thresh_ratio * self.frame_w

        self.good_frames_counter = 0
        self.last_message = ""
        self.captured = False
        self.debug_frame = None  # For visualizing detection steps

    def find_document_contour(self, frame):
        """
        Finds the document contour using a robust Canny/Contour pipeline.

        Args:
            frame (np.array): The input color frame.

        Returns:
            np.array: The 4-point contour (or None if not found).
        """
        try:
            # 1. Preprocessing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, self.gaussian_blur_kernel, 0)

            # 2. --- NEW PIPELINE: Canny Edge Detection ---
            # This is much better at finding outlines
            edges = cv2.Canny(blur, self.canny_thresh_1, self.canny_thresh_2)

            # 3. Dilate and Erode to close gaps in the edges
            dilate = cv2.dilate(edges, None, iterations=self.dilate_iterations)
            erode = cv2.erode(dilate, None, iterations=self.erode_iterations)

            # Save for debugging
            self.debug_frame = erode

            # 4. Find Contours
            # Use RETR_EXTERNAL to only find the outermost contours
            contours, _ = cv2.findContours(erode, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            # 5. Sort by area and find the largest 4-sided shape
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            for c in contours:
                # 6. Approximate the contour
                peri = cv2.arcLength(c, True)
                epsilon = self.approx_poly_epsilon_ratio * peri
                approx = cv2.approxPolyDP(c, epsilon, True)

                # 7. Check if it's a 4-sided quadrilateral
                if len(approx) == 4:
                    # It's a 4-sided shape. Now check its properties.

                    # Check 1: Min Area (to filter out tiny noise)
                    if cv2.contourArea(approx) < (self.frame_area * 0.01):
                        continue

                    # Check 2: Aspect Ratio
                    rect = cv2.minAreaRect(approx)
                    (x, y), (width, height), angle = rect

                    if width < height:
                        width, height = height, width

                    if height == 0:
                        continue

                    aspect_ratio = width / height

                    if abs(aspect_ratio - self.target_aspect_ratio) < self.aspect_ratio_tolerance:
                        # Found a 4-sided shape with the right aspect ratio!
                        return approx

            return None
        except Exception as e:
            print(f"Error in find_document_contour: {e}")
            return None

    def check_blur(self, frame):
        """
        Checks if the frame is blurry using the variance of the Laplacian.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = lap_var < self.blur_thresh
        return is_blurry, lap_var

    def check_glare(self, frame, doc_contour):
        """
        Checks for glare (specular highlights) within the document contour.
        """
        try:
            mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
            cv2.fillPoly(mask, [doc_contour], 255)
            doc_area = np.sum(mask == 255)
            if doc_area == 0:
                return False, 0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, glare_mask = cv2.threshold(gray, self.glare_thresh_val, 255, cv2.THRESH_BINARY)
            glare_in_doc = cv2.bitwise_and(glare_mask, mask)
            glare_pixels = np.sum(glare_in_doc == 255)
            glare_percentage = glare_pixels / doc_area
            has_glare = glare_percentage > self.glare_thresh_percent

            return has_glare, glare_percentage
        except Exception as e:
            print(f"Error in check_glare: {e}")
            return False, 0

    def check_position_and_size(self, doc_contour):
        """
        Checks if the document is centered and within size thresholds.
        """
        doc_area = cv2.contourArea(doc_contour)
        area_ratio = doc_area / self.frame_area

        if area_ratio < self.min_area_ratio:
            return "Move document closer"
        if area_ratio > self.max_area_ratio:
            return "Move document farther"

        M = cv2.moments(doc_contour)
        if M["m00"] == 0:
            return "Center document"

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        doc_center = (cx, cy)
        dist = np.linalg.norm(np.array(self.frame_center) - np.array(doc_center))

        if dist > self.center_thresh_pixels:
            return "Center document"

        return "ALL_CHECKS_PASSED"

    def order_points(self, pts):
        """
        Orders the 4 contour points into top-left, top-right,
        bottom-right, bottom-left.
        """
        rect = np.zeros((4, 2), dtype="float32")
        pts = pts.reshape(4, 2)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def perspective_transform(self, frame, doc_contour):
        """
        Applies a perspective transform to get a top-down view.
        """
        try:
            rect = self.order_points(doc_contour)
            (tl, tr, br, bl) = rect

            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxHeight = max(int(heightA), int(heightB))

            # --- Use target aspect ratio for a cleaner warp ---
            # This ensures the output is not 'squished'
            if maxHeight > 0:
                doc_aspect_ratio = maxWidth / maxHeight

                if doc_aspect_ratio > self.target_aspect_ratio:
                    # Too wide, adjust height
                    maxHeight = int(maxWidth / self.target_aspect_ratio)
                else:
                    # Too tall, adjust width
                    maxWidth = int(maxHeight * self.target_aspect_ratio)

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

            return warped
        except Exception as e:
            print(f"Error in perspective_transform: {e}")
            return None

    def draw_feedback(self, frame, message, contour=None):
        """
        Draws feedback text and contours on the frame.
        """
        if contour is not None:
            overlay = frame.copy()
            cv2.drawContours(overlay, [contour], -1, (0, 255, 0), cv2.FILLED)
            alpha = 0.1
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        else:
            # Draw the visual guide box
            x1 = int(self.frame_w * 0.1)
            y1 = int(self.frame_h * 0.1)
            x2 = int(self.frame_w * 0.9)
            y2 = int(self.frame_h * 0.9)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

        text_size, _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        text_w, text_h = text_size
        text_x = (self.frame_w - text_w) // 2
        text_y = (self.frame_h + text_h) // 2

        cv2.rectangle(frame, (text_x - 10, text_y - text_h - 10),
                      (text_x + text_w + 10, text_y + 10),
                      (0, 0, 0), -1)

        color = (0, 255, 0)
        if message != "Hold still...":
            color = (255, 255, 255)
        if message == "Avoid reflections":
            color = (0, 255, 255)
        if "closer" in message or "farther" in message or "Center" in message or "blurry" in message or "Place document" in message:
            color = (0, 0, 255)

        cv2.putText(frame, message, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

        return frame

    def run(self):
        """
        Main loop to process video frames and provide feedback.
        """
        while not self.captured:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            feedback_frame = frame.copy()

            # --- USE NEW LOGIC ---
            doc_contour = self.find_document_contour(frame)
            message = ""

            if doc_contour is None:
                message = "Place document in frame"
                self.good_frames_counter = 0
            else:
                # We found a document, now run quality checks
                pos_msg = self.check_position_and_size(doc_contour)

                if pos_msg != "ALL_CHECKS_PASSED":
                    message = pos_msg
                    self.good_frames_counter = 0
                else:
                    is_blurry, _ = self.check_blur(frame)
                    has_glare, _ = self.check_glare(frame, doc_contour)

                    if is_blurry:
                        message = "Too blurry"
                        self.good_frames_counter = 0
                    elif has_glare:
                        message = "Avoid reflections"
                        self.good_frames_counter = 0
                    else:
                        message = "Hold still..."
                        self.good_frames_counter += 1

            self.last_message = message
            feedback_frame = self.draw_feedback(feedback_frame, message, doc_contour)
            cv2.imshow("Document Scanner", feedback_frame)

            # Show debug window
            if self.debug_frame is not None:
                cv2.imshow("Debug - Edges", self.debug_frame)  # Renamed window

            if self.good_frames_counter > self.capture_frames_thresh:
                print(f"All checks passed for {self.capture_frames_thresh} frames. Capturing...")
                warped_doc = self.perspective_transform(frame, doc_contour)

                # if warped_doc is not None:
                #     save_path = "scanned_document.jpg"
                #     cv2.imwrite(save_path, warped_doc)
                #     print(f"Document saved to {save_path}")

                #     cv2.imshow("Scanned Document", warped_doc)
                #     self.captured = True
                #     cv2.waitKey(5000)
                if warped_doc is not None:

                    os.makedirs("captures", exist_ok=True)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    original_path = f"captures/original_{timestamp}.jpg"
                    warped_path = f"captures/warped_{timestamp}.jpg"

                    cv2.imwrite(original_path, frame)
                    cv2.imwrite(warped_path, warped_doc)

                    print(f"Original image saved to: {original_path}")
                    print(f"Warped image saved to: {warped_path}")

                    cv2.imshow("Scanned Document", warped_doc)

                    self.captured = True
                    cv2.waitKey(5000)
                else:
                    print("Could not capture and warp image.")
                    self.good_frames_counter = 0

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):  # 'r' to reset
                self.captured = False
                self.good_frames_counter = 0
                cv2.destroyWindow("Scanned Document")

        self.cap.release()
        cv2.destroyAllWindows()


# --- Main execution ---
if __name__ == "__main__":
    try:
        # ------------------------------------------------------------------
        # --- THIS IS THE MAIN TUNING SECTION ---
        # ------------------------------------------------------------------

        scanner = DocumentScanner(

            # --- CONTOUR DETECTION TUNING ---

            # (1.586 = ID Card), (1.4 ~ 1.414 = A4/A5 paper)
            target_aspect_ratio=1.586,
            aspect_ratio_tolerance=0.15,  # 15% tolerance

            # Kernel for blurring. (5,5) or (7,7) is good. Must be odd.
            gaussian_blur_kernel=(7, 7),

            # --- NEW CANNY PARAMETERS ---
            # Tune these if edges are not detected well.
            # Lower canny_thresh_1 (e.g., 30) to find weaker edges.
            # Raise canny_thresh_2 (e.g., 200) to reduce noise.
            canny_thresh_1=50,
            canny_thresh_2=150,

            # Adjust these to connect broken edges
            dilate_iterations=2,
            erode_iterations=1,

            # Simplifies the contour.
            # HIGHER (e.g., 0.04): More tolerance, can find "rounder" corners.
            # LOWER (e.g., 0.01): Stricter, needs sharp corners.
            approx_poly_epsilon_ratio=0.02,

            # --- QUALITY & POSITIONING CHECKS ---

            blur_thresh=45.0,
            glare_thresh_val=240,
            glare_thresh_percent=0.005,
            min_area_ratio=0.2,
            max_area_ratio=0.8,
            center_thresh_ratio=0.08,
            capture_frames_thresh=30
        )

        scanner.run()

    except Exception as e:
        print(f"An error occurred:{e}")