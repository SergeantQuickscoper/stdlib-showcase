/**
* @license Apache-2.0
*
* Copyright (c) 2025 The Stdlib Authors.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

'use strict';

// MODULES //

var fs = require( 'fs' );
var newton = require( '../lib/main.js' );
var Plot = require( '@stdlib/plot/ctor' );
var ndarray = require( '@stdlib/ndarray' );
var Float64Array = require( '@stdlib/array/float64' );

// Initial guess
var array2d = ndarray.array;
var initialX = array2d( new Float64Array( [ 0, 0 ] ), {
    shape: [ 2 ]
} );


// Newton's Method options 
var options = {
    tolerance: 1e-6,
    maxIterations: 1, // perform one step per call to visualize each iteration
    useLineSearch: true
};


// Track optimization path coordinates
var x = [];
var y = [];

var iterations = 0;
var maxSteps = 5000;


// Run iterative optimization
while ( iterations < maxSteps ) {
    // Get result of one iteration of the Rosenbrock function
    var result = newton.newtonsMethod( newton.rosenbrock, initialX, options );


    // Save new coordinates for plot
    x.push( result.solution.get( 0 ) );
    y.push( result.solution.get( 1 ) );

    if ( result.converged ) {
        break;
    }

    // Prepare for next iteration
    initialX = array2d( new Float64Array( [ result.solution.get( 0 ), result.solution.get( 1 ) ] ), {
        shape: [ 2 ]
    } );

    iterations++;
}


// Create plot to visualize the optimization path
var plot = new Plot( [ x ], [ y ], {
    title: 'Optimization Path with Newton\'s Method',
    xMin: 0,
    xMax: 10,
    yMin: 0,
    yMax: 10
} );


// Output HTML plot string into path.svg
var svg = plot.render( 'html' ); 
fs.writeFileSync( 'path.svg', svg, 'utf8' );
console.log( 'SVG saved to path.svg in the visualization folder' );
