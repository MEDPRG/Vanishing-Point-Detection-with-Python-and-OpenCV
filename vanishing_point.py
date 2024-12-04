import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt


def intersection_point_cramer(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if denominator == 0:
        return None  # Parallel lines

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    return (px, py)


def distance_point_to_line(line, point):
    """Calculate perpendicular distance from a point to a line segment."""
    x1, y1, x2, y2 = line
    px, py = point

    # Line segment length
    line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if line_length == 0:
        return float('inf')  # Avoid division by zero for degenerate lines

    # Distance formula
    distance = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / line_length
    return distance


def Ransac(lines, sigma):
    candidate_p = []
    current_inliers = []

    if lines is not None and len(lines) >= 2:
        for i in range((len(lines)) // 2):
            selected_line1, selected_line2 = random.sample(list(lines), 2)
            intersect = intersection_point_cramer(selected_line1[0], selected_line2[0])
            if intersect is not None:
                candidate_p.append(intersect)
    print(len(candidate_p))
    for point in candidate_p:
        init_inlier = []
        for j in range(len(lines)):
            if distance_point_to_line(lines[j][0], point) < sigma:
                init_inlier.append(lines[j][0])
        if init_inlier:
            current_inliers.append((point, init_inlier))
    print(".")
    # Find the point with the maximum number of init_inliers
    if current_inliers:
        Best_model = max(current_inliers, key=lambda x: len(x[1]))
        best_point, best_inliers = Best_model
        print(f"Point with maximum inliers: {best_point}")
        print(f"Number of inliers: {len(best_inliers)}")
        return best_point, best_inliers
    else:
        return []  # Return an empty list if no inliers found


# Re-estimate the vanishing point based on inliers using least-squares
def restimate_vanishing_point(inliers):
    A = []
    B = []

    for line in inliers:
        x1, y1, x2, y2 = line
        # Convert line endpoints to line equation in the form ax + by = c
        a = y2 - y1
        b = x1 - x2
        c = a * x1 + b * y1
        A.append([a, b])
        B.append(c)

    # Convert to numpy arrays
    A = np.array(A)
    B = np.array(B)

    # Use least-squares to solve for the best intersection point
    vp_x, vp_y = np.linalg.lstsq(A, B, rcond=None)[0]
    return (int(vp_x), int(vp_y))


def extend_line_to_vanishing_point(line, vp_x, vp_y, canvas_width, canvas_height):
    x1, y1, x2, y2 = line

    line_vector = np.array([x2 - x1, y2 - y1], dtype=float)
    line_vector /= np.linalg.norm(line_vector)  # Normalize the direction vector

    # Determine the extended endpoints that reach the vanishing point on the left
    if line_vector[0] != 0:
        t1 = (x1 - vp_x) / line_vector[0]
        t2 = (canvas_width - x1) / line_vector[0]
    else:
        t1, t2 = 0, 0

    pt1 = (int(x1 + t1 * 1000 * line_vector[0]), int(y1 + t1 * 1000 * line_vector[1]))
    pt2 = (int(x1 + t2 * 1000 * line_vector[0]), int(y1 + t2 * 1000 * line_vector[1]))

    return pt1, pt2


# Function to draw lines and the vanishing point on a black canvas with the image on the right side
def draw_result_on_black_canvas(image, vanishing_point, inliers):
    img_height, img_width = image.shape[:2]
    canvas_width = img_width

    # Vanishing point coordinates on the left side of the canvas
    vp_x, vp_y = int(vanishing_point[0]), int(vanishing_point[1])
    if vp_x <= 0:
        canvas_width = img_width * 2
        canvas = np.zeros((img_height, canvas_width, 3), dtype=np.uint8)  # Black canvas
        canvas[:, img_width:] = image
        vp = canvas_width - img_width + vp_x
        for inlier_line in inliers:
            x1, y1, x2, y2 = inlier_line
            inlier_line[0] = x1 + canvas_width / 2
            inlier_line[2] = x2 + canvas_width / 2
            pt1, pt2 = extend_line_to_vanishing_point(inlier_line, vp, vp_y, canvas_width, img_height)
            cv.line(canvas, (vp, vp_y), pt2, (0, 0, 255), 1)
            cv.line(canvas, pt1, (vp, vp_y), (0, 0, 255), 1)
        cv.circle(canvas, (vp, vp_y), 7, (0, 255, 0), -1)
    elif vp_x >= img_width:
        canvas_width = img_width * 2
        canvas = np.zeros((img_height, canvas_width, 3), dtype=np.uint8)  # Black canvas
        canvas[:, :img_width] = image
        vp = canvas_width - img_width + (vp_x - img_width)
        for inlier_line in inliers:
            pt1, pt2 = extend_line_to_vanishing_point(inlier_line, vp, vp_y, canvas_width, img_height)
            cv.line(canvas, pt1, pt2, (0, 0, 255), 1)
        cv.circle(canvas, (vp, vp_y), 7, (0, 255, 0), -1)
    else:
        canvas = image.copy()
        # Draw all lines in red on the image portion
        # Draw inlier lines in green
        for inlier_line in inliers:
            pt1, pt2 = extend_line_to_vanishing_point(inlier_line, vp_x, vp_y, canvas_width, img_height)
            cv.line(canvas, pt1, pt2, (0, 0, 255), 1)
        cv.circle(canvas, (vp_x, vp_y), 7, (0, 255, 0), -1)
    return canvas


if __name__ == "__main__":
    # Load the image
    img2 = cv.resize(cv.imread('path/image.png'), (700, 700))
    img = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    lsd = cv.createLineSegmentDetector(cv.LSD_REFINE_STD)

    seg_lines, width, precision, nfa = lsd.detect(img)
    seg_lines2 = seg_lines.copy()

    # Draw the detected lines segment
    lines_img = img2.copy()
    lsd.drawSegments(lines_img, seg_lines)
    # lsd.drawSegments(img2, seg_lines)
    sigm = 10
    print('sigm; ', sigm)
    best_vanishing_point, best_inliers = Ransac(seg_lines, sigm)  # Get all inliers
    print(f'Number of current inliers: {len(best_inliers)}')
    print(best_vanishing_point)
    print(best_inliers)

    if best_inliers:  # Check if there are any inliers
        # re-estimate the vanishing point
        vanishing_point = restimate_vanishing_point(best_inliers)
    else:
        vanishing_point = None

    print(f'vanishing_point:{vanishing_point}')
    if vanishing_point:
        # Convert vanishing point coordinates to integers
        vp_x, vp_y = int(vanishing_point[0]), int(vanishing_point[1])

        for line in best_inliers:
            x1, y1, x2, y2 = line
            cv.line(img2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    # Draw inlier lines and vanishing point
    result_canvas = draw_result_on_black_canvas(img2, vanishing_point, best_inliers)

    # Display the result
    cv.imshow('LSD', lines_img)
    cv.imshow("Vanishing Point with Lines1", result_canvas)

    # change the name or the path if you want to save it and uncommented it
    # cv.imwrite("ELTECar_images/ELTECar2_Vanishing_point.png", result_canvas)
    cv.waitKey(0)
    cv.destroyAllWindows()
