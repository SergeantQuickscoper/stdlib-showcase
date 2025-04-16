# Newton's Method for Real Valued Function Optimization

A JavaScript implementation of Newton's method for unconstrained numerical optimization using the [stdlib](https://stdlib.io/) library. This project is a showcase for GSoc 2025, of my proposal for the implementation of [Idea #27 Optimization Algorithms](https://github.com/stdlib-js/google-summer-of-code/issues/27)

## Overview

Newton's method is a second-order optimization technique that uses both gradient and Hessian information to iteratively find the minimum of a function. This implementation features:

- Numerical gradient and Hessian computation via finite differences
- A Linear System solver using `stdlib` BLAS for computing Newton steps
- Testing across various function types

## stdlib Integration and showcase

This project heavily utilizes `stdlib` packages for computing:

- **@stdlib/array**: For typed arrays (Float64Array)
- **@stdlib/ndarray**: For n-dimensional array manipulation
- **@stdlib/blas**: For basic linear algebra operations
- **@stdlib/math**: For mathematical functions and operations

The implementation demonstrates how `stdlib` can be used as a foundation for building advanced numerical algorithms with JavaScript.

## Implementation Details

### Core Components (main.js)

1. **Numerical Differentiation**
   - `numericalGradient`: Computes function gradients using central finite differences
   - `numericalHessian`: Computes the Hessian matrix (second derivatives)

2. **Linear System Solver**
   - Implementation for solving the Newton step equation (Hessian * step = -gradient)
   - Includes Cramer's rule for 2×2 systems and Gaussian elimination for larger systems

3. **Optimization Algorithm Implemented**
   - `newtonsMethod`: The main optimization function implementing Newton's method
   - Configurable parameters: tolerance, maximum iterations, step size, line search

## Test File (test.js)

The test file verifies the correctness and performance of the implementation through:

1. **Function Optimization Tests**
   - Simple quadratic functions with known minima
   - Rosenbrock function and Himmelblau function (optimization benchmark)
   - Bowl function with multiple starting points (testing robustness)
   - Exponential functions (non-quadratic behavior)


2. **Numerical Derivative Tests**
   - Gradient computation accuracy
   - Hessian matrix computation accuracy

The tests focus on 2D optimization problems which are easier to visualize and represent common use cases. The Rosenbrock function in particular demonstrates the algorithm's ability to handle challenging optimization landscapes with narrow curved valleys.

## Usage Example

```javascript
// Create initial point
var initialX = array2d( new Float64Array([0, 0]), { 'shape': [2] } );

// Options for the optimizer
var options = {
    'tolerance': 1e-6,
    'maxIterations': 100,
    'useLineSearch': true
};

// Run optimization
var result = newtonsMethod( targetFunction, initialX, options );

console.log( 'Solution:', [result.solution.get(0), result.solution.get(1)] );
console.log( 'Function value:', result.value );
console.log( 'Iterations:', result.iterations );
console.log( 'Converged:', result.converged );
```

## Installation and Running

```bash
# Install dependencies
npm install

# Run tests
npm test

# Run example optimization
npm start

# Run visualization of the Rosenbrock algorithm optimized from (0, 0)
npm visualize
```

## Visualizer with `stdlib/plot`

The plot.js file offers a visualization of Newton's Method applied to the Rosenbrock function, starting from an initial guess of 0,0 in SVG format using `stdlib/plot`. To generate your own custom plots, modify the following parameters:
- **Initial Guess**: To modify the starting poinnt of the plot.
- **Test Funcntion**: To change the used Test function between the Rosenbrock and Himmelblau functions.
- **Plot API Parameters**: To modify the look and scale of the resultant plot.

Then simply run ```npm visualize``` to generate the SVG output file.

## Conclusion

This project demonstrates how stdlib provides a foundation for implementing optimization algorithms in JavaScript with clean and maintainable code.

