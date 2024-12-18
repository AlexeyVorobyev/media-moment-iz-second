from abc import ABC, abstractmethod


class MatrixOperator(ABC):
    @abstractmethod
    def x_matrix(self, img, x, y) -> int:
        ...

    @abstractmethod
    def y_matrix(self, img, x, y) -> int:
        ...
    

class ScharrOperator(MatrixOperator):
    def x_matrix(self, img, x, y):
        """
        Применение оператора Щарра для нахождения Gx
        Матрица выглядит следующим образом:
        -3 0 3
        -10 0 10
        -3 0 3
        :param img: Исходное изображение
        :param x: Координата пикселя по X
        :param y: Координата пикселя по Y
        :return:
        """
        return -3 * int(img[x - 1][y - 1]) - 10 * int(img[x][y - 1]) - 3 * int(img[x + 1][y - 1]) + \
            3 * int(img[x - 1][y + 1]) + 10 * int(img[x][y + 1]) + 3 * int(img[x + 1][y + 1])

    def y_matrix(self, img, x, y):
        """
        Применение оператора Щарра для нахождения Gy
        Матрица выглядит следующим образом:
        -3 -10 -3
        0 0 0
        3 10 3
        :param img: Исходное изображение
        :param x: Координата пикселя по X
        :param y: Координата пикселя по Y
        :return:
        """
        return -3 * int(img[x - 1][y - 1]) - 10 * int(img[x - 1][y]) - 3 * int(img[x - 1][y + 1]) + \
            3 * int(img[x + 1][y - 1]) + 10 * int(img[x + 1][y]) + 3 * int(img[x + 1][y + 1])


class RobertsOperator(MatrixOperator):

    def x_matrix(self, img, x, y):
        """
        Применение оператора Робертса для нахождения Gx
        Матрица выглядит следующим образом:
        1 0
        0 -1
        :param img: Исходное изображение
        :param x: Координата пикселя по X
        :param y: Координата пикселя по Y
        :return:
        """
        return img[x][y] - int(img[x + 1][y - 1])

    def y_matrix(self, img, x, y):
        """
        Применение оператора Робертса для нахождения Gy
        Матрица выглядит следующим образом:
        0 1
        -1 0
        :param img: Исходное изображение
        :param x: Координата пикселя по X
        :param y: Координата пикселя по Y
        :return:
        """
        return img[x + 1][y] - int(img[x][y + 1])


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
