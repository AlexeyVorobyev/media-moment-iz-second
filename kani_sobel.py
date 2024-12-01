import cv2
from kani_algorytm import KaniAlgorythm, ImageShowKaniAlgorythmEnum

images = [
    {
        "path": "resources/original/apple.png",
        "name": "kani_apple",
    },
    {
        "path": "resources/original/jeleps.png",
        "name": "kani_jeleps",
    },
    {
        "path": "resources/original/linux.png",
        "name": "kani_linux",
    },
    {
        "path": "resources/original/person.png",
        "name": "kani_person",
    }
]

deviations = [
    10, 1, 0.1
]

threshold_dividers = [
    (40, 2), (20, 5), (15, 10)
]

for deviation in deviations:
    for threshold_divider in threshold_dividers:
        kaniAlgo = KaniAlgorythm(
            image_size=(500, 500),
            image_show_list=[],
            kernel_size=7,
            deviation=deviation,
            threshold_dividers=threshold_divider,
        )

        for image in images:
            processed_image, _ = kaniAlgo.process_image_with_return(image["path"])

            cv2.imwrite(
                f"resources/kani_sobel/{image["name"]}_gauss_{deviation}_threshold_{threshold_divider[0]}_{threshold_divider[1]}.png",
                processed_image
            )

cv2.waitKey(0)
cv2.destroyAllWindows()