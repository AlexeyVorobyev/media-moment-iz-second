from abc import ABC, abstractmethod


class MatrixOperator(ABC):
    @abstractmethod
    def x_matrix(self, img, x, y):
        raise NotImplementedError

    @abstractmethod
    def y_matrix(self, img, x, y):
        raise NotImplementedError


class SobelOperator(MatrixOperator):

    def x_matrix(self, img, x, y):
        """
        Применение оператора Собеля для нахождения Gx
        Матрица выглядит следующим образом:
        -1 0 1
        -2 0 2
        -1 0 1
        :param img: Исходное изображение
        :param x: Координата пикселя по X
        :param y: Координата пикселя по Y
        :return:
        """
        return -int(img[x - 1][y - 1]) - 2 * int(img[x][y - 1]) - int(img[x + 1][y - 1]) + \
            int(img[x - 1][y + 1]) + 2 * int(img[x][y + 1]) + int(img[x + 1][y + 1])

    def y_matrix(self, img, x, y):
        """
        Применение оператора Собеля для нахождения Gy
        Матрица выглядит следующим образом:
        -1 -2 -1
        0 0 0
        1 2 1
        :param img: Исходное изображение
        :param x: Координата пикселя по X
        :param y: Координата пикселя по Y
        :return:
        """
        return -int(img[x - 1][y - 1]) - 2 * int(img[x - 1][y]) - int(img[x - 1][y + 1]) + \
            int(img[x + 1][y - 1]) + 2 * int(img[x + 1][y]) + int(img[x + 1][y + 1])