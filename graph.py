import cv2

from graph_algorytm import GraphAlgorythm, ImageShowGraphAlgorythmEnum

images = [
    {
        "path": "resources/original/apple.png",
        "name": "graph_apple",
    },
    {
        "path": "resources/original/jeleps.png",
        "name": "graph_jeleps",
    },
    {
        "path": "resources/original/linux.png",
        "name": "graph_linux",
    },
    {
        "path": "resources/original/person.png",
        "name": "graph_person",
    }
]

deviations = [
    10, 1, 0.1
]

size_component_thresholds = [
    0, 100, 150
]

contrasts = [
    10, 15, 20
]

for deviation in deviations:
    for size_component_threshold in size_component_thresholds:
        for contrast in contrasts:
            graphAlgo = GraphAlgorythm(
                image_size=(500, 500),
                image_show_list=[
                    # ImageShowGraphAlgorythmEnum.CONTRAST
                ],
                kernel_size=7,
                deviation=deviation,
                size_component_threshold=size_component_threshold,
                contrast=contrast,
            )

            for image in images:
                processed_image = graphAlgo.process_image_with_return(image["path"])

                cv2.imwrite(
                    f"resources/graph/{image["name"]}_gauss_{deviation}_size_component_threshold_{size_component_threshold}_contrast{contrast}.png",
                    processed_image
                )

cv2.waitKey(0)
cv2.destroyAllWindows()