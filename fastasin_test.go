package tdigest

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFastAsin(t *testing.T) {
	tests := []struct {
		name    string
		in      float64
		outTest func(float64) bool
	}{
		{"neg", -1, func(x float64) bool { return x == -1.5707963267948966 }},
		{"nan", 4, math.IsNaN},
		{"neg", 0.9, func(x float64) bool { return x == 1.1197695149986342 }},
	}
	for _, test := range tests {
		tt := test
		t.Run(tt.name, func(t *testing.T) {
			assert.True(t, tt.outTest(fastAsin(tt.in)))
		})
	}
}
