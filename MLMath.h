#ifndef __MACHINE_LEARNING_MATH_H__
#define __MACHINE_LEARNING_MATH_H__

#include <cmath>

typedef double Float;
typedef Float (*RealFunc)(Float);
typedef RealFunc ActivationFunc;
typedef RealFunc ActivationDirFunc;

inline Float Sigmoid(Float x)
{
	return 1 / (1 + exp(-x));
}
inline Float SigmoidDir(Float x)
{
	Float fPow = exp(-x);
	Float fBase = fPow + 1;
	return fPow / (fBase * fBase);
}
inline Float SigmoidDirOutput(Float x)
{
	return (1 - x) * x;
}

inline Float Tanh(Float x)
{
	Float xneg = exp(-x);
	Float xpos = exp(x);
	return (xpos - xneg) / (xpos + xneg);
}
inline Float TanhDir(Float x)
{
	Float out = Tanh(x);
	return 1 - out * out;
}
inline Float TanhDirOutput(Float x)
{
	return 1 - x * x;
}

inline Float ReLU(Float x)
{
    if (x >= 0)
        return x;
    return 0;
}
inline Float ReLUDir(Float x)
{
    if (x >= 0)
        return 1;
    return 0;
}




#endif //__MACHINE_LEARNING_MATH_H__