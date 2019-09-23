package tdigest_test

import (
	"testing"

	"fmt"

	"github.com/google/go-cmp/cmp"
	"github.com/signalfx/tdigest"
	"github.com/stretchr/testify/assert"
)

func TestCentroid_Add(t *testing.T) {
	tests := []struct {
		name    string
		c       tdigest.Centroid
		r       tdigest.Centroid
		want    tdigest.Centroid
		wantErr bool
		errStr  string
	}{
		{
			name: "error when weight is zero",
			r: tdigest.Centroid{
				Weight: -1.0,
			},
			wantErr: true,
			errStr:  "centroid weight cannot be less than zero",
		},
		{
			name: "zero weight",
			c: tdigest.Centroid{
				Weight: 0.0,
				Mean:   1.0,
			},
			r: tdigest.Centroid{
				Weight: 1.0,
				Mean:   2.0,
			},
			want: tdigest.Centroid{
				Weight: 1.0,
				Mean:   2.0,
			},
		},
		{
			name: "weight order of magnitude",
			c: tdigest.Centroid{
				Weight: 1,
				Mean:   1,
			},
			r: tdigest.Centroid{
				Weight: 10,
				Mean:   10,
			},
			want: tdigest.Centroid{
				Weight: 11,
				Mean:   9.181818181818182,
			},
		},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			c := &tt.c
			if err := c.Add(tt.r); (err != nil) != tt.wantErr {
				t.Errorf("Centroid.Add() error = %v, wantErr %v", err, tt.wantErr)
			} else if tt.wantErr && err.Error() != tt.errStr {
				t.Errorf("Centroid.Add() error.Error() = %s, errStr %v", err.Error(), tt.errStr)
			}
			if !cmp.Equal(tt.c, tt.want) {
				t.Errorf("unexprected centroid -want/+got\n%s", cmp.Diff(tt.want, tt.c))
			}
		})
	}
}

func TestNewCentroidList(t *testing.T) {
	tests := []struct {
		name      string
		centroids []tdigest.Centroid
		want      tdigest.CentroidList
	}{
		{
			name: "empty list",
		},
		{
			name: "priority should be by mean ascending",
			centroids: []tdigest.Centroid{
				{
					Mean: 2.0,
				},
				{
					Mean: 1.0,
				},
			},
			want: tdigest.CentroidList{
				{
					Mean: 1.0,
				},
				{
					Mean: 2.0,
				},
			},
		},
		{
			name: "single element should be identity",
			centroids: []tdigest.Centroid{
				{
					Mean: 1.0,
				},
			},
			want: tdigest.CentroidList{
				{
					Mean: 1.0,
				},
			},
		},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			if got := tdigest.NewCentroidList(tt.centroids); !cmp.Equal(tt.want, got) {
				t.Errorf("NewCentroidList() = -want/+got %s", cmp.Diff(tt.want, got))
			}
		})
	}
}

func TestCentroid_String(t *testing.T) {
	c := tdigest.Centroid{Weight: 1, Mean: 7}
	assert.Equal(t, c.String(), "{mean: 7.000000 weight: 1.000000}")
}

func TestCentroidListSerde(t *testing.T) {
	c := tdigest.NewWithCompression(50)
	for i := 0; i < 100; i++ {
		c.Add(float64(i), 1)
	}
	cc := make(tdigest.CentroidList, len(c.Centroids()))
	copy(cc, c.Centroids())
	bb, err := cc.MarshalBinary()
	fmt.Println(bb)
	assert.Nil(t, err)
	c1 := tdigest.NewWithCompression(50)
	err = tdigest.UnmarshalCentroidListIntoTdigest(bb, c1)
	assert.Nil(t, err)
	assert.Equal(t, len(c.Centroids()), len(c1.Centroids()))
	for i, v := range c1.Centroids() {
		assert.Equal(t, v, cc[i])
	}
	fmt.Println(cc)
}
