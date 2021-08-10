#pragma once

#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>

void gauleg(int n, double x[], double w[]);
void gauher(int n, double x[], double w[]);
void gauher2(int n, double x[], double w[]);

void SPDRegularization(double SPDMatrix[], int dimension, double criteria);

void newtonStep(double gradient[], double hessian[], int dimension, double newtonStepArray[]);