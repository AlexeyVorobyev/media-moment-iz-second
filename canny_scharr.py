import cv2
from canny_algorithm import CannyAlgorithm
from matrix_operators import RobertsOperator, ScharrOperator

images = [
    {
        "path": "resources/original/image-1.jpg",
        "name": "kanni_sharr_samokat_1",
    },
    {
        "path": "resources/original/image-2.jpeg",
        "name": "kanni_sharr_samokat_2",
    },
    {
        "path": "resources/original/image-3.jpg",
        "name": "kanni_sharr_samokat_3",
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
        canny_alg = CannyAlgorithm(
            image_size=(500, 500),
            image_show_list=[],
            kernel_size=7,
            matrix_operator=ScharrOperator(),
            deviation=deviation,
            threshold_dividers=threshold_divider,
        )

        for image in images:
            processed_image, _ = canny_alg.process_image_with_return(image["path"])

            cv2.imwrite(
                f"resources/kani_sharr/{image["name"]}_gauss_{deviation}_threshold_{threshold_divider[0]}_{threshold_divider[1]}.png",
                processed_image
            )

cv2.waitKey(0)
cv2.destroyAllWindows()
