import cv2
import os
import glob
import hashlib

def find_best_fingerprint_match(query_img_path: str,
                                folder_path: str,
                                match_distance_threshold: int = 50):
    """
    Returns the single best matching fingerprint in `folder_path` to
    the `query_img_path`, using ORB + brute‐force matching.
    
    Returns:
      (best_filepath: str, best_score: int)
    """
    # 1) Load query in grayscale
    query = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE)
    if query is None:
        raise FileNotFoundError(f"Cannot load query image at {query_img_path}")

    # 2) ORB detector + BF matcher
    orb = cv2.ORB_create()
    kp_q, des_q = orb.detectAndCompute(query, None)
    if des_q is None:
        raise RuntimeError("No features found in query image")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 3) Iterate candidates, dedupe by hash
    best_score = -1
    best_path = None
    seen_hashes = set()

    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        for path in glob.glob(os.path.join(folder_path, ext)):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # dedupe identical images
            h = hashlib.md5(img.tobytes()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            kp_c, des_c = orb.detectAndCompute(img, None)
            if des_c is None:
                continue

            # match and count “good” ones
            matches = bf.match(des_q, des_c)
            good = [m for m in matches if m.distance < match_distance_threshold]
            score = len(good)

            if score > best_score:
                best_score, best_path = score, path

    if best_path is None:
        raise RuntimeError("No matching fingerprints found")

    return best_path, best_score


# # Example usage
# if __name__ == "__main__":
#     query_image = "reconstructed_fingerprint.jpg"
#     folder = "original_fingerprints"
#     best_file, best_score = find_best_fingerprint_match(query_image, folder)
#     print(f"Best match: {best_file} (score={best_score})")
