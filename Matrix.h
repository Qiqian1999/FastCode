#ifndef DEF_MATRIX
#define DEF_MATRIX

#include <vector>
#include <iostream>
#include <stdexcept>
#include "Utility.h"

class Matrix
{
public:
	Matrix();
	Matrix(int height, int width);
    Matrix(int height, int width, int padding);
	Matrix(std::vector<std::vector<double>> const &array);
    Matrix(std::vector<std::vector<double>> const &array, int padding);
	
	//2D vector stores matrix
	std::vector<std::vector<double>> matrix;

	//functions
	int getIndexValue(int i, int j) const;
	int getHeight() const;
	int getWidth() const;
	int dotProduct(Matrix &other) const;
	int getMax() const;
	void checkIfEqual(Matrix &other) const;
	void add(Matrix other);
	void print() const;
	Matrix filterSlide(Matrix filter, int stride, int bias);
    Matrix filterSlideSimd(Matrix filter1, Matrix filter2, Matrix filter3, Matrix filter4, int stride, int bias);
    double* singleElement(Matrix filter1, Matrix filter2, Matrix filter3, Matrix filter4, int startX, int startY);
	Matrix filterSlide(Matrix filter, int stride, int bias, int padding);
	Matrix maxSlide(int H, int F, int stride, int bias);

	std::vector<std::vector<double>> getPadMatrix(int padding);

private:
	int height;
	int width;
	int padding;
	void padMatrix();
};

#endif 
