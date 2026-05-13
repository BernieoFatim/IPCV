import cv2
import numpy as np

# Load images (grayscale)
img1 = cv2.imread("target.jpg", 0)
img2 = cv2.imread("scene.jpg", 0)

# SIFT
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN matcher
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# Lowe ratio test
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

# Draw matches
result = cv2.drawMatches(img1, kp1, img2, kp2, good, None)

cv2.imshow("Matches", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(good) > 10:

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # Find homography using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Get size of target image
    h, w = img1.shape

    # Define corners of target image
    pts = np.float32([
        [0,0],
        [0,h],
        [w,h],
        [w,0]
    ]).reshape(-1,1,2)

    # Transform corners to scene
    dst = cv2.perspectiveTransform(pts, H)

    # Draw polygon on scene image
    scene_color = cv2.imread("scene.jpg")

    cv2.polylines(
        scene_color,
        [np.int32(dst)],
        True,
        ((0, 255, 0)),
        3
    )

    cv2.imshow("Detected Book", scene_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Load overlay image
overlay = cv2.imread("overlay.jpg")

# Resize overlay to match target size
overlay = cv2.resize(overlay, (w, h))

# Warp overlay into scene
warped = cv2.warpPerspective(
    overlay,
    H,
    (scene_color.shape[1], scene_color.shape[0])
)

# Create mask
mask = np.zeros(
    (scene_color.shape[0], scene_color.shape[1]),
    dtype=np.uint8
)

cv2.fillPoly(mask, [np.int32(dst)], 255)

mask_inv = cv2.bitwise_not(mask)

# Black-out the area of book in original image
bg = cv2.bitwise_and(scene_color, scene_color, mask=mask_inv)

# Take only overlay region
fg = cv2.bitwise_and(warped, warped, mask=mask)

# Combine
final = cv2.add(bg, fg)

cv2.imshow("AR Overlay", final)
cv2.waitKey(0)
cv2.destroyAllWindows()