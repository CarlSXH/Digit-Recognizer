#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <cassert>
#include <cmath>
#include <cstring>
#include <Windows.h>

typedef unsigned int UInt;

namespace Matrices
{

template<typename T>
struct Matrix
{
    UInt Row, Col;
    T *mf;

    Matrix() : 
        Row(0), Col(0), mf(0)
    {
    }

    T& operator()(UInt n)
    {
        return mf[n];
    }
    const T& operator()(UInt n) const
    {
        return mf[n];
    }

    T& operator()(UInt r, UInt c)
    {
        return mf[r*Col + c];
    }
    const T& operator()(UInt r, UInt c) const
    {
        return mf[r*Row + c];
    }
};

template<typename T>
UInt MatIndex(const Matrix<T> *m, const UInt r, const UInt c)
{
    assert(Valid(m));
    return r * m->Col + c;
}

template<typename T>
bool Valid(const Matrix<T> *p)
{
    return p != 0 && p->mf != NULL;
}

template<typename T>
void MatInit(Matrix<T> *m, UInt row, UInt column)
{
    assert(m != 0);

    m->mf = new T[row * column];
    m->Row = row;
    m->Col = column;
}
template<typename T>
void MatDelete(Matrix<T> *m)
{
    assert(m != 0);

    if (m->mf != 0)
    {
        delete[] m->mf;
        m->mf = 0;
    }
    m->Row = 0;
    m->Col = 0;
}

template<typename T>
void MatClear(Matrix<T> *m, const T v)
{
    assert(Valid(m));

    for (UInt i = 0; i < m->Row * m->Col; i++)
        m->mf[i] = v;
}
template<typename T>
void MatRnd(Matrix<T> *m, const T _min, const T _max)
{
    assert(Valid(m));
    assert(_max > _min);

    for (UInt i = 0; i < m->Row * m->Col; i++)
        m->mf[i] = rand() * (T)(3.0518509475997192297128208258309e-5f) * (_max - _min) + _min;
}
template<typename T>
void MatCpy(Matrix<T> *m, const Matrix<T> *a)
{
    assert(Valid(m) && Valid(a));
    assert(MatSameSize(m, a));

    for (UInt i = 0; i < m->Row * m->Col; i++)
        m->mf[i] = a->mf[i];
}
template<typename T>
void MatRegion(Matrix<T> *m, const Matrix<T> *a, const UInt r, const UInt c, const UInt x=0, const UInt y=0)
{
    assert(Valid(m) && Valid(a));
    assert(c == m->Col && r == m->Row);
    assert(c+x <= a->Col && r+y <= a->Row);

    for (UInt i = 0; i < r; i++)
        for (UInt j = 0; j < c; j++)
            m->mf[MatIndex(m, i, j)] = a->mf[MatIndex(a, i+y, j+x)];
}


template<typename T>
void MatDebug(const Matrix<T> *m)
{
    for (int i = 0; i < m->Row; i++)
    {
        for (int j = 0; j < m->Col; j++)
        {
            std::string s = std::to_string(m->mf[MatIndex(m, i, j)]);
            OutputDebugString(s.c_str());
            OutputDebugString(" ");
        }
        OutputDebugString("\n");
    }
}

template<typename T>
bool MatSameSize(const Matrix<T> *a, const Matrix<T> *b)
{
    assert(Valid(a) && Valid(b));
    return a->Row == b->Row && a->Col == b->Col;
}

template<typename T>
bool MatDotable(const Matrix<T> *a, const Matrix<T> *b)
{
    assert(Valid(a) && Valid(b));
    return a->Col == b->Row;
}

template<typename T>
void MatApply(Matrix<T> *out, const Matrix<T> *a, T (CalcFunc)(T))
{
    assert(Valid(out) && Valid(a));
    assert(MatSameSize(out, a));

    for (UInt i = 0; i < a->Row * a->Col; i++)
        out->mf[i] = CalcFunc(a->mf[i]);
}
template<typename T>
void MatApply(Matrix<T> *out, const Matrix<T> *a, T t, T (CalcFunc)(T, T))
{
	assert(Valid(out) && Valid(a));
	assert(MatSameSize(out, a));

	for (UInt i = 0; i < a->Row * a->Col; i++)
		out->mf[i] = CalcFunc(a->mf[i], t);
}
template<typename T>
void MatApply(Matrix<T> *out, const Matrix<T> *a, const Matrix<T> *b, T (CalcFunc)(T, T))
{
    assert(Valid(out) && Valid(a) && Valid(b));
    assert(MatSameSize(out, a) && MatSameSize(a, b));

    for (UInt i = 0; i < a->Row * a->Col; i++)
        out->mf[i] = CalcFunc(a->mf[i], b->mf[i]);
}
template<typename T>
void MatApply(Matrix<T> *out, const Matrix<T> *a, const Matrix<T> *b, T t, T (CalcFunc)(T, T, T))
{
	assert(Valid(out) && Valid(a) && Valid(b));
	assert(MatSameSize(out, a) && MatSameSize(a, b));

	for (UInt i = 0; i < a->Row * a->Col; i++)
		out->mf[i] = CalcFunc(a->mf[i], b->mf[i], t);
}
template<typename T>
void MatApply(T *out, const Matrix<T> *a, T (CalcFunc)(T))
{
    assert(out != NULL && Valid(a));

    T result = (T)0;
    for (int i = 0; i < a->Col * a->Row; i++)
        result += CalcFunc(a->mf[i]);

    *out = result;
}

template<typename T>
void MatAdd(Matrix<T> *out, const Matrix<T> *a, const Matrix<T> *b)
{
    assert(Valid(out) && Valid(a) && Valid(b));
    assert(MatSameSize(a, b) && MatSameSize(out, a));

    for (UInt i = 0; i < a->Row * a->Col; i++)
        out->mf[i] = a->mf[i] + b->mf[i];
}
template<typename T>
void MatSub(Matrix<T> *out, const Matrix<T> *a, const Matrix<T> *b)
{
    assert(Valid(out) && Valid(a) && Valid(b));
    assert(MatSameSize(a, b) && MatSameSize(out, a));

    for (UInt i = 0; i < a->Row * a->Col; i++)
        out->mf[i] = a->mf[i] - b->mf[i];
}
template<typename T>
void MatMul(Matrix<T> *out, const Matrix<T> *a, const Matrix<T> *b)
{
    assert(Valid(out) && Valid(a) && Valid(b));
    assert(MatSameSize(a, b) && MatSameSize(out, a));

    for (UInt i = 0; i < a->Row * a->Col; i++)
        out->mf[i] = a->mf[i] * b->mf[i];
}
template<typename T>
void MatDiv(Matrix<T> *out, const Matrix<T> *a, const Matrix<T> *b)
{
    assert(Valid(out) && Valid(a) && Valid(b));
    assert(MatSameSize(a, b) && MatSameSize(out, a));

    for (UInt i = 0; i < a->Row * a->Col; i++)
        out->mf[i] = a->mf[i] / b->mf[i];
}

template<typename T>
void MatMul(Matrix<T> *out, const Matrix<T> *a, const T b)
{
    assert(Valid(out) && Valid(a));
    assert(MatSameSize(out, a));

    for (UInt i = 0; i < a->Row * a->Col; i++)
        out->mf[i] = a->mf[i] * b;
}
template<typename T>
void MatAdd(Matrix<T> *out, const Matrix<T> *a, const T b)
{
    assert(Valid(out) && Valid(a));
    assert(MatSameSize(out, a));

    for (UInt i = 0; i < a->Row * a->Col; i++)
        out->mf[i] = a->mf[i] + b;
}
template<typename T>
void MatSub(Matrix<T> *out, const Matrix<T> *a)
{
    assert(Valid(out) && Valid(a));
    assert(MatSameSize(out, a));

    for (UInt i = 0; i < a->Row * a->Col; i++)
        out->mf[i] = -a->mf[i];
}

template<typename T>
void MatDot(Matrix<T> *out, const Matrix<T> *a, const Matrix<T> *b)
{
    assert(Valid(out) && Valid(a) && Valid(b));
    assert(MatDotable(a, b) && out->Row == a->Row && out->Col == b->Col);

    MatClear(out, (T)0);
    for (UInt i = 0; i < a->Row; i++)
        for (UInt j = 0; j < b->Col; j++)
            for (UInt k = 0; k < a->Col; k++)
                out->mf[MatIndex(out, i, j)] += a->mf[MatIndex(a, i, k)] * b->mf[MatIndex(b, k, j)];
}
template<typename T>
void MatDotDiag(Matrix<T> *out, const Matrix<T> *a, const Matrix<T> *b)
{
	assert(Valid(out) && Valid(a) && Valid(b));
	assert(a->Row == out->Row && b->Row == out->Col);
	assert(a->Col == b->Col);

	MatClear(out, (T)0);
	for (UInt i = 0; i < out->Row; i++)
		for (UInt j = 0; j < out->Col; j++)
			for (UInt k = 0; k < a->Col; k++)
				out->mf[MatIndex(out, i, j)] += a->mf[MatIndex(a, i, k)] * b->mf[MatIndex(b, j, k)];
}

template<typename T>
void MatT(Matrix<T> *out, const Matrix<T> *a)
{
    assert(Valid(out) && Valid(a));
    assert(out->Row == a->Col && out->Col == a->Row);
    for (int i = 0; i < out->Row; i++)
        for (int j = 0; j < out->Col; j++)
            out->mf[MatIndex(out, i, j)] = a->mf[MatIndex(a, j, i)];
}


}

//template<UInt Row>
//Matrix<Row, 1> MultiplyCol(Matrix<Row, 1> mMatrix, Matrix<Row, 1> mScale)
//{
//    Matrix<Row, 1> Result;
//    for (UInt i = 0; i < Row; i++)
//        Result.mf[i][0] = mMatrix.mf[i][0] * mScale.mf[i][0];
//    return Result;
//}
//
//template<UInt Col>
//Matrix<1, Col> MultiplyRow(Matrix<1, Col> mMatrix, Matrix<1, Col> mScale)
//{
//    Matrix<1, Col> Result;
//    for (UInt i = 0; i < Col; i++)
//        Result.mf[0][i] = mMatrix.mf[0][i] * mScale.mf[0][i];
//    return Result;
//}


#endif //__MATRIX_H__