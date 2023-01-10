#ifndef __NEURAL_NETWORK_H__
#define __NEURAL_NETWORK_H__

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>

#include <cfloat>
#include "Matrix.h"
#include "MLMath.h"


#ifdef _DEBUG
    //#define _CPU_DEBUG
    //#define _CPU_RUNTIME_DEBUG
#endif

using std::string;
using std::ifstream;
using std::ofstream;

//#define _USE_FLOAT
//#define _USE_DOUBLE
//
//#ifdef _USE_DOUBLE
//
//#ifdef _USE_FLOAT
//#error "Can't use both float and double"
//#endif
//
//typedef double Float;
//#define _FLOAT_MAX DBL_MAX
//#define _FLOAT_MIN DBL_MIN
//
//#undef _USE_DOUBLE
//
//#endif
//
//#ifdef _USE_FLOAT
//
//#ifdef _USE_DOUBLE
//#error "Can't use both float and double"
//#endif
//
//typedef float Float;
//#define _FLOAT_MAX FLT_MAX
//#define _FLOAT_MIN FLT_MIN
//
//#undef _USE_FLOAT
//
//#endif

typedef double Float;
const Float _FLOAT_MAX = DBL_MAX;
const Float _FLOAT_MIN = DBL_MIN;

typedef Matrices::Matrix<Float> Matrix;
typedef Matrices::Matrix<int> MatrixI;



inline Float RandRange(Float fMin, Float fMax);

inline Float SingleDir(Float a, Float b);
inline Float Power2(Float a);

struct ForwardResult
{
    // Vectors
    // m_nCount or (layer count) elements. Each element represent the output of the neurons of the layer.
    // The first element is for the first layer.
    // The size of the nth element is (layer count(n))
    Matrix *m_LayerAs;

    // Vectors
    // (m_nCount-1) or (layer count-1) elements. The nth element is the output of the neuron in layer (n+1) before the activation function
    // The first element is for the second layer
    // The size of the nth element is (layer count(n+1))
    Matrix *m_LayerZs;

    int m_nCount;
};
struct Input
{
    // Vector
    // (input count) elements.
    Matrix m_Input;
};
struct TargetResult
{
    // Vector
    // (output count) elements.
    Matrix m_Target;
};
struct ResultDerivative
{
    // Matrices
    // (layer count - 1) elements.
    Matrix *m_pDWeights;
    // Vectors
    // (layer count - 1) elements;
    Matrix *m_pDBiases;
};
struct DerivativeTemp
{
    // Vectors
    Matrix *m_Errors;
    Matrix *m_ErrorsT;
};
struct GradientCheckResult
{
    Float m_dMax, m_dMin, m_dAverage, m_dSum;
};

struct TrainingParam
{
    // The learning rate
    Float m_fAlpha;
    // The regularization parameter
    Float m_fLamda;


    int m_nSampleNum;
};
struct TrainTemp
{
    ResultDerivative m_AccuDerivative;

    ForwardResult m_ForwardResult;
    ResultDerivative m_Derivative;
    DerivativeTemp m_Temp;
};

struct Progression
{
    Float trainError, testError;
    Float alpha;
    int start, batchsize;
    int trainCount, testCount;
};

struct NeuralNetwork
{
    // Matrices
    // m_LayerCount.Row or (layer count - 1) elements.
    Matrix *m_pWeights;
    // Vectors
    // m_LayerCount.Row or (layer count - 1) elements.
    Matrix *m_pBiases;
    MatrixI m_LayerCount;

#ifdef _CPU_RUNTIME_DEBUG
    bool m_bDebug;
#endif

    NeuralNetwork();

    void Create(int *rpfLayerCount, int rnCount);

    // 4   bits, 32 bit int: magic number (0x0098f8ab) (completely random XD).
    // 4   bits, 32 bit int: number of layers (n).
    // 4*n bits, 32 bit int: the number of units in layers.
    // 1   bit,  8 bit char: 1 (true) or 0 (false), whether it's using float (1) or double (0)
    // ... bits, 32 or 64 bit floating number: weights.
    // ... bits, 32 or 64 bit floating number: bias.
    bool CreateFromFile(string fileName);
    void SaveToFile(string fileName);
    void Delete();

    void Initialize(ForwardResult *result);
    void Initialize(Input *input);
    void Initialize(TargetResult *target);
    void Initialize(ResultDerivative *derivative);
    void Initialize(DerivativeTemp *temp);
    void Initialize(GradientCheckResult *numerical);
    void Initialize(TrainTemp *temp);
    void Delete(ForwardResult *result);
    void Delete(Input *input);
    void Delete(TargetResult *target);
    void Delete(ResultDerivative *derivative);
    void Delete(DerivativeTemp *temp);
    void Delete(GradientCheckResult *numerical);
    void Delete(TrainTemp *temp);

    void Forward(Input input, ForwardResult *result, int n = 0);
    Float Cost(ForwardResult result, TargetResult target/*, Float sumSqr*/);
    Float SumSqr();

    void Derivative(ForwardResult result, TargetResult target, DerivativeTemp temp, ResultDerivative *derivative);


    void NumericalGradientCheck(Input input, TargetResult target, ResultDerivative derivative, GradientCheckResult *numerical, TrainingParam param);

    void Train(Input *inputs, TargetResult *targets, int count, TrainingParam param, TrainTemp trainTemp);
};


void __Main();

#endif //__NEURAL_NETWORK_H__




//#ifndef __NEURAL_NETWORK_H__
//#define __NEURAL_NETWORK_H__
//
//#include <iostream>
//#include <fstream>
//#include <string>
//#include <cmath>
//
//#include "eigen\Dense"
//#include <cfloat>
//
//namespace Network
//{
//
//using std::string;
//using std::ifstream;
//using std::ofstream;
//
////#define _USE_FLOAT
//#define _USE_DOUBLE
//
//#ifdef _USE_DOUBLE
//
//#ifdef _USE_FLOAT
//#error "Can't use both float and double"
//#endif
//typedef double Float;
//#define _FLOAT_MAX DBL_MAX
//#define _FLOAT_MIN DBL_MIN
//
//typedef Eigen::MatrixXd Matrix;
//typedef Eigen::MatrixXd Matrix;
//
//#endif
//
//#ifdef _USE_FLOAT
//
//#ifdef _USE_DOUBLE
//#error "Can't use both float and double"
//#endif
//
//typedef float Float;
//#define _FLOAT_MAX FLT_MAX
//#define _FLOAT_MIN FLT_MIN
//
//typedef Eigen::MatrixXf Matrix;
//typedef Eigen::MatrixXf Matrix;
//
//#endif
//
//typedef Eigen::MatrixXi MatrixI;
//typedef Eigen::MatrixXi MatrixI;
//
//inline Float Sigmoid(Float fInput)
//{
//    return 1 / (1 + exp(-fInput));
//}
//// Derivative of Sigmoid
//inline Float SigmoidDir(Float fInput)
//{
//    Float fPow = exp(-fInput);
//    Float fBase = fPow + 1;
//    return fPow / (fBase * fBase);
//    // return pow(M_E, -fInput) / powf((1 + powf(M_E, -fInput)), 2);
//}
//
//inline Float RandRange(Float fMin, Float fMax)
//{
//    return rand() * 3.0518509475997192297128208258309e-5f * (fMax - fMin) + fMin;
//}
//
//struct ForwardResult
//{
//    // m_nCount or (layer count) elements. Each element represent the output of the neurons of the layer.
//    // The first element is for the first layer.
//    // The size of the nth element is (layer count(n))
//    Matrix *m_LayerAs;
//
//    // (m_nCount-1) or (layer count-1) elements. The nth element is the output of the neuron in layer (n+1) before the activation function
//    // The first element is for the second layer
//    // The size of the nth element is (layer count(n+1))
//    Matrix *m_LayerZs;
//
//    int m_nCount;
//};
//struct Input
//{
//    // (input count) elements.
//    Matrix m_Input;
//};
//struct TargetResult
//{
//    // (output count) elements.
//    Matrix m_Target;
//};
//struct ResultDerivative
//{
//    // (layer count - 1) elements.
//    Matrix *m_pDWeights;
//    // (layer count - 1) elements;
//    Matrix *m_pDBiases;
//};
//struct DerivativeTemp
//{
//    Matrix *m_Errors;
//    Float *m_BiasErrors;
//};
//struct GradientCheckResult
//{
//    Float m_dMax, m_dMin, m_dAverage, m_dSum;
//};
//
//struct NeuralNetwork
//{
//    // m_LayerCount.Row or (layer count - 1) elements.
//    Matrix *m_pWeights;
//    // m_LayerCount.Row or (layer count - 1) elements.
//    Matrix *m_pBiases;
//    MatrixI m_LayerCount;
//
//
//    NeuralNetwork() :
//        m_pWeights(0), m_LayerCount()
//    {
//    }
//
//    void Create(int *rpfLayerCount, int rnCount)
//    {
//        m_LayerCount = MatrixI(rnCount);
//        for (int i = 0; i < rnCount; i++)
//            m_LayerCount(i) = rpfLayerCount[i];
//
//        m_pWeights = new Matrix[rnCount - 1];
//        m_pBiases = new Matrix[rnCount - 1];
//        for (int i = 0; i < rnCount - 1; i++)
//        {
//            m_pWeights[i] = Matrix::Random(m_LayerCount(i + 1), m_LayerCount(i)) * 0.5 + Matrix::Constant(m_LayerCount(i + 1), m_LayerCount(i), 0.5);
//            m_pBiases[i] = Matrix::Random(m_LayerCount(i + 1)) * 0.5 + Matrix::Constant(m_LayerCount(i + 1), 0.5);
//        }
//    }
//
//    // 4   bits, 32 bit int: magic number (0x0098f8ab) (completely random XD).
//    // 4   bits, 32 bit int: number of layers (n).
//    // 4*n bits, 32 bit int: the number of units in layers.
//    // 1   bit,  8 bit char: 1 (true) or 0 (false), whether it's using float (1) or double (0)
//    // ... bits, 32 or 64 bit floating number: weights.
//    // ... bits, 32 or 64 bit floating number: bias.
//    void CreateFromFile(string fileName)
//    {
//        ifstream fin(fileName, std::ios::binary);
//
//        int magicNum = 0x0098f8ab;
//        fin.read((char*)(&magicNum), 4);
//
//        int layerNum;
//        fin.read((char*)(&layerNum), 4);
//
//        m_LayerCount = MatrixI(layerNum);
//        m_pWeights = new Matrix[layerNum - 1];
//        m_pBiases = new Matrix[layerNum - 1];
//
//        for (int i = 0; i < layerNum; i++)
//        {
//            int unitNum = m_LayerCount(i);
//            fin.read((char*)(&unitNum), 4);
//            m_LayerCount(i) = unitNum;
//        }
//        for (int i = 0; i < layerNum - 1; i++)
//        {
//            m_pWeights[i] = Matrix(m_LayerCount(i + 1), m_LayerCount(i));
//            m_pBiases[i] = Matrix(m_LayerCount(i + 1));
//        }
//
//        char typeFlag;
//
//        fin.read(&typeFlag, 1);
//
//        int size = 0;
//        if (typeFlag == 0)
//            size = sizeof(double);
//        else if (typeFlag == 1)
//            size = sizeof(float);
//
//        // temporary variable
//        Float f;
//
//        for (int i = 0; i < layerNum - 1; i++)
//        {
//            for (int j = 0; j < m_pWeights[i].Row; j++)
//            {
//                for (int k = 0; k < m_pWeights[i].Col; k++)
//                {
//                    fin.read((char*)(&f), size);
//                    m_pWeights[i](j, k) = f;
//                }
//            }
//        }
//
//        for (int i = 0; i < layerNum - 1; i++)
//        {
//            for (int j = 0; j < m_pBiases[i].Row; j++)
//            {
//                fin.read((char*)(&f), size);
//                m_pBiases[i](j) = f;
//            }
//        }
//    }
//    void SaveToFile(string fileName)
//    {
//        ofstream fout(fileName, std::ios::out | std::ios::binary | std::ios::ate);
//
//        int magicNum = 0x0098f8ab;
//        fout.write((char*)(&magicNum), 4);
//
//        int layerNum = m_LayerCount.Row;
//        fout.write((char*)(&layerNum), 4);
//
//        for (int i = 0; i < layerNum; i++)
//        {
//            int unitNum = m_LayerCount(i);
//            fout.write((char*)(&unitNum), 4);
//        }
//
//
//        char typeFlag;
//
//#ifdef _USE_FLOAT
//        typeFlag = 1;
//#endif
//#ifdef _USE_DOUBLE
//        typeFlag = 0;
//#endif
//
//        fout.write(&typeFlag, 1);
//
//        // temporary variable
//        Float f;
//
//        for (int i = 0; i < layerNum - 1; i++)
//        {
//            for (int j = 0; j < m_pWeights[i].Row; j++)
//            {
//                for (int k = 0; k < m_pWeights[i].Col; k++)
//                {
//                    f = m_pWeights[i](j, k);
//                    fout.write((char*)(&f), sizeof(Float));
//                }
//            }
//        }
//
//        for (int i = 0; i < layerNum - 1; i++)
//        {
//            for (int j = 0; j < m_pBiases[i].Row; j++)
//            {
//                f = m_pBiases[i](j);
//                fout.write((char*)(&f), sizeof(Float));
//            }
//        }
//    }
//    void Delete()
//    {
//        if (m_pWeights != NULL)
//        {
//            delete[] m_pWeights;
//            m_pWeights = NULL;
//        }
//        if (m_pBiases != NULL)
//        {
//            delete[] m_pBiases;
//            m_pBiases = NULL;
//        }
//    }
//
//    void Initialize(ForwardResult *result)
//    {
//        result->m_nCount = m_LayerCount.Row;
//        result->m_LayerAs = new Matrix[result->m_nCount];
//        result->m_LayerZs = new Matrix[result->m_nCount - 1];
//    }
//    void Initialize(Input *input)
//    {
//    }
//    void Initialize(TargetResult *target)
//    {
//    }
//    void Initialize(ResultDerivative *derivative)
//    {
//        derivative->m_pDWeights = new Matrix[m_LayerCount.Row - 1];
//        derivative->m_pDBiases = new Matrix[m_LayerCount.Row - 1];
//    }
//    void Initialize(DerivativeTemp *temp)
//    {
//        temp->m_Errors = new Matrix[m_LayerCount.Row - 1];
//        temp->m_BiasErrors = new Float[m_LayerCount.Row - 1];
//    }
//    void Initialize(GradientCheckResult *numerical)
//    {
//    }
//    void Delete(ForwardResult *result)
//    {
//        if (result->m_LayerAs != NULL)
//        {
//            delete[] result->m_LayerAs;
//            result->m_LayerAs = NULL;
//        }
//        if (result->m_LayerZs != NULL)
//        {
//            delete[] result->m_LayerZs;
//            result->m_LayerZs = NULL;
//        }
//        result->m_nCount = 0;
//    }
//    void Delete(Input *input)
//    {
//    }
//    void Delete(TargetResult *result)
//    {
//    }
//    void Delete(ResultDerivative *result)
//    {
//        if (result->m_pDWeights != NULL)
//        {
//            delete[] result->m_pDWeights;
//            result->m_pDWeights = NULL;
//        }
//        if (result->m_pDBiases != NULL)
//        {
//            delete[] result->m_pDBiases;
//            result->m_pDBiases = NULL;
//        }
//    }
//    void Delete(DerivativeTemp *temp)
//    {
//        if (temp->m_Errors != NULL)
//        {
//            delete[] temp->m_Errors;
//            temp->m_Errors = NULL;
//        }
//        if (temp->m_BiasErrors != NULL)
//        {
//            delete[] temp->m_BiasErrors;
//            temp->m_BiasErrors = NULL;
//        }
//    }
//    void Delete(GradientCheckResult *numerical)
//    {
//    }
//
//    void Forward(Input input, ForwardResult *result)
//    {
//        result->m_nCount = m_LayerCount.Row;
//        result->m_LayerAs[0] = Matrix(input.m_Input);
//
//        for (int i = 1; i < m_LayerCount.Row; i++)
//        {
//            result->m_LayerZs[i - 1] = m_pWeights[i - 1] * result->m_LayerAs[i - 1];
//            result->m_LayerZs[i - 1] += m_pBiases[i - 1];
//            result->m_LayerAs[i] = result->m_LayerZs[i - 1].unaryExpr(&Sigmoid);
//        }
//    }
//    Float Cost(ForwardResult result, TargetResult target)
//    {
//        Matrix x = result.m_LayerAs[result.m_nCount - 1];
//        Matrix y = target.m_Target;
//        Float sum = 0;
//
//        for (int i = 0; i < x.Row; i++)
//        {
//            if (y(i) == 0)
//                sum += log(1 - x(i));
//            else
//                sum += log(x(i));
//        }
//
//        sum = -sum;
//        return sum;
//    }
//    void Derivative(ForwardResult result, TargetResult target, DerivativeTemp temp, ResultDerivative *derivative)
//    {
//        int count = m_LayerCount.Row;
//        Matrix *errors = temp.m_Errors;
//
//        errors[count - 2] = result.m_LayerAs[result.m_nCount - 1] - target.m_Target;
//
//        for (int i = count - 3; i >= 0; i--)
//        {
//            Matrix v1 = m_pWeights[i + 1].transpose() * errors[i + 1];
//            Matrix v2 = result.m_LayerZs[i].unaryExpr(&SigmoidDir);
//            errors[i] = (v1).array() * v2.array();
//        }
//
//        for (int i = 0; i < count - 1; i++)
//        {
//            derivative->m_pDWeights[i] = errors[i] * (result.m_LayerAs[i].transpose());
//            derivative->m_pDBiases[i] = Matrix(errors[i]);
//        }
//    }
//    void NumericalGradientCheck(Input input, TargetResult target, ResultDerivative derivative, GradientCheckResult *numerical)
//    {
//        Float epsilon = 0.0001;
//        Float sum = 0.0;
//        Float max = -_FLOAT_MAX, min = _FLOAT_MAX;
//
//        ForwardResult result;
//
//        Initialize(&result);
//
//        for (int i = 0; i < m_LayerCount.Row - 1; i++)
//        {
//            for (int j = 0; j < m_pWeights[i].Row; j++)
//            {
//                for (int k = 0; k < m_pWeights[i].Col; k++)
//                {
//                    Float dir = 0.0;
//
//                    m_pWeights[i](j, k) += epsilon;
//                    Forward(input, &result);
//                    dir = Cost(result, target);
//
//                    m_pWeights[i](j, k) -= 2 * epsilon;
//                    Forward(input, &result);
//                    dir = dir - Cost(result, target);
//
//                    m_pWeights[i](j, k) += epsilon;
//
//                    dir = dir / (epsilon * 2);
//
//                    Float dif = dir - derivative.m_pDWeights[i](j, k);
//                    if (dif > max)
//                        max = dif;
//                    if (dif < min)
//                        min = dif;
//                    sum += dif;
//                }
//            }
//        }
//
//        for (int i = 0; i < m_LayerCount.Row - 1; i++)
//        {
//            for (int j = 0; j < m_pBiases[i].Row; j++)
//            {
//                Float dir = 0.0;
//
//                m_pBiases[i](j) += epsilon;
//                Forward(input, &result);
//                dir = Cost(result, target);
//
//                m_pBiases[i](j) -= 2 * epsilon;
//                Forward(input, &result);
//                dir = dir - Cost(result, target);
//
//                m_pBiases[i](j) += epsilon;
//
//                dir = dir / (epsilon * 2);
//
//                Float dif = dir - derivative.m_pDBiases[i](j);
//                if (dif > max)
//                    max = dif;
//                if (dif < min)
//                    min = dif;
//                sum += dif;
//            }
//        }
//
//        Delete(&result);
//
//        int count = 0;
//
//        for (int i = 0; i < m_LayerCount.Row - 1; i++)
//        {
//            count += m_pWeights[i].size();
//            count += m_pBiases[i].size();
//        }
//
//        numerical->m_dMax = max;
//        numerical->m_dMin = min;
//        numerical->m_dSum = sum;
//        numerical->m_dAverage = sum / count;
//    }
//
//    void Train(Input *inputs, TargetResult *targets, int count, Float alpha)
//    {
//        ResultDerivative accuDerivative;
//
//        ForwardResult forwardResult;
//        ResultDerivative derivative;
//        DerivativeTemp temp;
//
//        Initialize(&accuDerivative);
//
//        Initialize(&forwardResult);
//        Initialize(&derivative);
//        Initialize(&temp);
//
//        Float cost = 0;
//
//        for (int i = 0; i < m_LayerCount.Row - 1; i++)
//        {
//            accuDerivative.m_pDWeights[i] = Matrix::Zero(m_pWeights[i].Row, m_pWeights[i].Col);
//            accuDerivative.m_pDBiases[i] = Matrix::Zero(m_pBiases[i].Row);
//        }
//
//        do
//        {
//            cost = 0;
//            for (int i = 0; i < m_LayerCount.Row - 1; i++)
//            {
//                accuDerivative.m_pDWeights[i].setZero();
//                accuDerivative.m_pDBiases[i].setZero();
//            }
//
//            for (int i = 0; i < count; i++)
//            {
//                Forward(inputs[i], &forwardResult);
//                Derivative(forwardResult, targets[i], temp, &derivative);
//                cost += Cost(forwardResult, targets[i]);
//                for (int i = 0; i < m_LayerCount.Row - 1; i++)
//                {
//                    accuDerivative.m_pDWeights[i] += derivative.m_pDWeights[i];
//                    accuDerivative.m_pDBiases[i] += derivative.m_pDBiases[i];
//                }
//            }
//
//            for (int i = 0; i < m_LayerCount.Row - 1; i++)
//            {
//                accuDerivative.m_pDWeights[i] /= (Float)count;
//                accuDerivative.m_pDBiases[i] /= (Float)count;
//
//                m_pWeights[i] -= accuDerivative.m_pDWeights[i] * alpha;
//                m_pBiases[i] -= accuDerivative.m_pDBiases[i] * alpha;
//            }
//
//            cost /= count;
//        } while (cost >= 0.01);
//
//
//
//
//        Delete(&forwardResult);
//        Delete(&derivative);
//        Delete(&temp);
//    }
//};
//
//
//void Main()
//{
//    NeuralNetwork net;
//    int layers[5] = { 28 * 28, 15, 10 };
//    net.Create(layers, 3);
//
//    Input v;
//
//    v.m_Input = Matrix::Random(28 * 28) * 0.5 + Matrix::Constant(28 * 28, 0.5);
//
//    ForwardResult result;
//    net.Initialize(&result);
//
//    net.Forward(v, &result);
//
//    TargetResult target;
//    net.Initialize(&target);
//
//    target.m_Target = Matrix::Random(10);
//
//    for (int i = 0; i < 10; i++)
//    {
//        target.m_Target(i) = RandRange(0, 2) > 1.0 ? 1 : 0;
//    }
//
//    ResultDerivative derivative;
//    net.Initialize(&derivative);
//
//    DerivativeTemp temp;
//    net.Initialize(&temp);
//
//    for (int i = 0; i < 1000; i++)
//        net.Forward(v, &result);
//
//    GradientCheckResult numerical;
//
//    net.NumericalGradientCheck(v, target, derivative, &numerical);
//
//    net.Delete(&result);
//    net.Delete(&target);
//    net.Delete(&derivative);
//    net.Delete(&temp);
//
//    net.Delete();
//
//    return;
//}
//
//}
//
//
//
//#endif //__NEURAL_NETWORK_H__