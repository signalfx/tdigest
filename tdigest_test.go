package tdigest

import (
	"testing"

	"fmt"
	"math"
	"reflect"
	"strconv"
	"time"

	"os"
	"strings"

	"github.com/stretchr/testify/assert"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
)

const (
	N     = 1e6
	Mu    = 10
	Sigma = 3

	seed = 42
)

var (
	// NormalData is a slice of N random values that are normally distributed with mean Mu and standard deviation Sigma.
	NormalData           []float64
	UniformData          []float64
	benchmarkCompression = float64(50)
	benchmarkMedian      = float64(2.675264e09)
	benchmarkStdDev      = float64(13.14254e09)
	benchmarkDecayValue  = 0.9
	benchmarkDecayEvery  = int32(1000)
	benchmarks           = []struct {
		name  string
		scale scaler
	}{
		//{name: "k1", scale: &K1{}},
		//{name: "k1_fast", scale: &K1Fast{}},
		//{name: "k1_spliced", scale: &K1Spliced{}},
		//{name: "k1_spliced_fast", scale: &K1SplicedFast{}},
		//{name: "k2", scale: &K2{}},
		//{name: "k2_spliced", scale: &K2Spliced{}},
		//{name: "k3", scale: &K3{}},
		{name: "k3_spliced", scale: &K3Spliced{}},
		//{name: "kquadratic", scale: &KQuadratic{}},
	}
)

func init() {
	dist := distuv.Normal{
		Mu:    Mu,
		Sigma: Sigma,
		Src:   rand.New(rand.NewSource(seed)),
	}
	uniform := rand.New(rand.NewSource(seed))

	UniformData = make([]float64, N)

	NormalData = make([]float64, N)

	for i := range NormalData {
		NormalData[i] = dist.Rand()
		UniformData[i] = uniform.Float64() * 100
	}
}

func TestTdigest_Quantile(t *testing.T) {
	tests := []struct {
		name     string
		data     []float64
		quantile float64
		want     float64
		epsilon  float64
	}{
		{name: "increasing", quantile: 0.5, data: []float64{1, 2, 3, 4, 5}, want: 3},
		{name: "data in decreasing order", quantile: 0.25, data: []float64{555.349107, 432.842597}, want: 432.842597},
		{name: "small", quantile: 0.5, data: []float64{1, 2, 3, 4, 5, 5, 4, 3, 2, 1}, want: 3},
		{name: "small 99 (max)", quantile: 0.99, data: []float64{1, 2, 3, 4, 5, 5, 4, 3, 2, 1}, want: 5},
		{name: "normal 50", quantile: 0.5, data: NormalData, want: 10.000744215323294, epsilon: 0.000124},
		{name: "normal 90", quantile: 0.9, data: NormalData, want: 13.841895725158281, epsilon: 3.6343e-05},
		{name: "uniform 50", quantile: 0.5, data: UniformData, want: 49.992136904768316, epsilon: 0.000315},
		{name: "uniform 90", quantile: 0.9, data: UniformData, want: 89.98220402280788, epsilon: 4.6e-05},
		{name: "uniform 99", quantile: 0.99, data: UniformData, want: 98.98511738020078, epsilon: 1.4e-05},
		{name: "uniform 99.9", quantile: 0.999, data: UniformData, want: 99.90131708898765, epsilon: 8.279e-06},
	}
	for _, tt := range tests {
		tt := tt
		for _, bt := range benchmarks {
			bt := bt
			t.Run(tt.name+"-"+bt.name, func(t *testing.T) {
				td := NewWithScaler(bt.scale, 1000, 0, 0)
				for _, x := range tt.data {
					td.Add(x, 1)
				}
				if td.CheckWeights() > 0 {
					t.Errorf("unexpected checkweights result; %d > 9", td.CheckWeights())
				}
				got := td.Quantile(tt.quantile)
				actual := quantile(tt.quantile, tt.data)
				assert.InEpsilon(t, tt.want, got, tt.epsilon, "unexpected quantile %f, got %g want %g", tt.quantile, got, tt.want)
				assert.InEpsilon(t, actual, got, tt.epsilon, "unexpected quantile %f, got %g want %g", tt.quantile, got, tt.want)
			})
		}
	}
}

func TestClear(t *testing.T) {
	in1 := simpleTDigest(0)
	in2 := simpleTDigest(1)
	in3 := simpleTDigest(10000)
	in2.Clear()
	in3.Clear()
	tests := []struct {
		name  string
		left  *TDigest
		right *TDigest
	}{
		{"one", in1, in2},
		{"two", in2, in3},
		{"three", in1, in3},
	}
	testcase := func(left *TDigest, right *TDigest) {
		if !reflect.DeepEqual(left, right) {
			t.Errorf("clearning round trip resulted in changes")
			t.Logf("inn: %+v", left)
			t.Logf("out: %+v", right)
		}
	}
	for _, test := range tests {
		tt := test
		t.Run(test.name, func(t *testing.T) {
			testcase(tt.left, tt.right)
		})
	}
}

func TestClone(t *testing.T) {
	testcase := func(in *TDigest) func(*testing.T) {
		return func(t *testing.T) {
			out := in.Clone()
			if !reflect.DeepEqual(in, out) {
				t.Errorf("marshaling round trip resulted in changes")
				t.Logf("in: %+v", in)
				t.Logf("out: %+v", out)
			}
		}
	}
	t.Run("empty", testcase(New()))
	t.Run("1 value", testcase(simpleTDigest(1)))
	t.Run("1000 values", testcase(simpleTDigest(1000)))

	d := New()
	d.Add(1, 1)
	d.Add(1, 1)
	d.Add(0, 1)
	t.Run("1, 1, 0 input", testcase(d))
}

func TestTdigest_CDFs(t *testing.T) {
	tests := []struct {
		name    string
		data    []float64
		cdf     float64
		want    float64
		epsilon float64
	}{
		{name: "increasing", cdf: 3, data: []float64{1, 2, 3, 4, 5}, want: 0.5},
		{name: "small", cdf: 4, data: []float64{1, 2, 3, 4, 5, 5, 4, 3, 2, 1}, want: 0.7, epsilon: 0.072},
		{name: "small max", cdf: 5, data: []float64{1, 2, 3, 4, 5, 5, 4, 3, 2, 1}, want: 0.9, epsilon: 0.12},
		{name: "normal mean", cdf: 10, data: NormalData, want: 0.499925, epsilon: 0.000221},
		{name: "normal high", cdf: -100, data: NormalData, want: 0},
		{name: "normal low", cdf: 110, data: NormalData, want: 1},
		{name: "uniform 50", cdf: 50, data: UniformData, want: 0.500068, epsilon: 0.000334},
		{name: "uniform min", cdf: 0, data: UniformData, want: 0},
		{name: "uniform max", cdf: 100, data: UniformData, want: 1},
		//{name: "uniform 10", cdf: 10, data: UniformData, want: 0.099872, epsilon: 0.000158},
		{name: "uniform 90", cdf: 90, data: UniformData, want: 0.900155, epsilon: 7.4018e-05},
		{name: "uniform 99", cdf: 99, data: UniformData, want: 0.990157, epsilon: 7.1891e-06},
	}
	for _, tt := range tests {
		tt := tt
		for _, bt := range benchmarks {
			bt := bt
			t.Run(tt.name+"-"+bt.name, func(t *testing.T) {
				td := NewWithScaler(bt.scale, 1000, 0, 0)
				for _, x := range tt.data {
					td.Add(x, 1)
				}
				if td.CheckWeights() > 0 {
					t.Errorf("unexpected checkweights result; %d > 9", td.CheckWeights())
				}
				got := td.CDF(tt.cdf)
				actual := cdf(tt.cdf, tt.data)
				if got != tt.want {
					assert.InEpsilon(t, tt.want, got, tt.epsilon, "unexpected CDF %f, got %g want %g", tt.cdf, got, tt.want)
				}
				if got != actual {
					assert.InEpsilon(t, actual, got, tt.epsilon, "unexpected CDF %f, got %g want %g", tt.cdf, got, actual)
				}
			})
		}
	}
}

func TestCloneRoundTrip(t *testing.T) {
	testcase := func(in *TDigest) func(*testing.T) {
		return func(t *testing.T) {

			out := in.Clone()
			if !reflect.DeepEqual(in, out) {
				t.Errorf("marshaling round trip resulted in changes")
				t.Logf("inn: %+v", in)
				t.Logf("out: %+v", out)
			}
		}
	}
	t.Run("empty", testcase(New()))
	t.Run("1 value", testcase(simpleTDigest(1)))
	t.Run("1000 values", testcase(simpleTDigest(1000)))

	d := New()
	d.Add(1, 1)
	d.Add(1, 1)
	d.Add(0, 1)
	t.Run("1, 1, 0 input", testcase(d))
}

func getVal() float64 {
	return math.Abs(rand.NormFloat64())*benchmarkStdDev + benchmarkMedian
}

func TestSizesVsCap(t *testing.T) {
	m := map[string]int{
		"k1":              100,
		"k1_fast":         100,
		"k1_spliced":      100,
		"k1_spliced_fast": 100,
		"k2":              100,
		"k2_spliced":      35,
		"k3":              100,
		"k3_spliced":      30,
		"kquadratic":      100,
	}
	for _, test := range benchmarks {
		test := test
		t.Run(test.name, func(t *testing.T) {
			td := NewWithScaler(test.scale, 50, 0, 0)
			n := 100000
			for i := 0; i < n; i++ {
				td.Add(getVal(), 1.0)
			}
			td.process()
			fmt.Printf("\t\t\t\t\t\tn: %d len: %d cap: %d %d\n", n, len(td.processed), cap(td.processed), cap(td.unprocessed))

			if len(td.processed) > m[test.name] {
				t.Errorf("unexpected centroid size %d > %d", len(td.processed), m[test.name])
			}
			if td.CheckWeights() > 0 {
				t.Errorf("unexpected checkweights result; %d > 9", td.CheckWeights())
			}
		})
	}
}

func BenchmarkMainAdd(b *testing.B) {
	rand.Seed(uint64(time.Now().Unix()))
	b.ReportAllocs()
	for _, bm := range benchmarks {
		bm := bm
		b.Run(bm.name, func(b *testing.B) {
			td := getTd(bm.scale, b)
			td.process()
		})
	}
}

func getTd(scale scaler, b *testing.B) *TDigest {
	td := NewWithScaler(scale, benchmarkCompression, benchmarkDecayValue, benchmarkDecayEvery)
	data := getData(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		td.Add(data[i], 1.0)
	}
	//if td.CheckWeights() > 0 {
	//	b.Errorf( "unexpected checkweights result; %d > 9", td.CheckWeights())
	//}
	return td
}

func BenchmarkMainQuantile(b *testing.B) {
	rand.Seed(uint64(time.Now().Unix()))
	for _, bm := range benchmarks {
		bm := bm
		b.Run(bm.name, func(b *testing.B) {
			td := getTd(bm.scale, b)
			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				td.Quantile(rand.Float64())
			}
		})
	}
}

func BenchmarkMainCDF(b *testing.B) {
	rand.Seed(uint64(time.Now().Unix()))
	for _, bm := range benchmarks {
		bm := bm
		b.Run(bm.name, func(b *testing.B) {
			td := getTd(bm.scale, b)
			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				td.CDF(getVal())
			}
		})
	}
}

func BenchmarkCompression(b *testing.B) {
	benchmarks := []struct {
		compression int
	}{
		{1000},
		{500},
		{250},
		{125},
		{100},
		{50},
	}
	for _, bm := range benchmarks {
		bm := bm
		b.Run("Compression "+strconv.Itoa(bm.compression), func(b *testing.B) {
			b.ReportAllocs()
			td := NewWithDecay(float64(bm.compression), benchmarkDecayValue, benchmarkDecayEvery)
			data := make([]float64, b.N)
			for i := 0; i < b.N; i++ {
				data[i] = getVal()
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				td.Add(data[i], 1.0)
			}
			q := td.Quantile(0.99)
			b.StopTimer()
			actual := quantile(0.99, data)
			fmt.Println("\n", "proc", len(td.processed), cap(td.processed), "unproc", cap(td.unprocessed), td.maxProcessed, q, actual, math.Abs(q-actual))
		})
	}
}

func BenchmarkMultipleHistos(b *testing.B) {
	benchmarks := []struct {
		name string
		size int64
	}{
		{name: "10", size: 10},
		{name: "100", size: 100},
		{name: "1000", size: 1000},
		{name: "10000", size: 10000},
		{name: "100000", size: 100000},
	}
	for _, bm := range benchmarks {
		bm := bm
		b.Run(bm.name+"-double", func(b *testing.B) {
			data := getData(b)
			b.ReportAllocs()
			td := NewWithDecay(benchmarkCompression, benchmarkDecayValue, benchmarkDecayEvery)
			td2 := NewWithDecay(benchmarkCompression, benchmarkDecayValue, benchmarkDecayEvery)
			for i := 0; i < b.N; i++ {
				td.Add(data[i], 1)
				td2.Add(data[i], 1)
				if int64(i)%bm.size == 0 {
					td2.Clear()
				}
			}
		})
		b.Run(bm.name+"-merge", func(b *testing.B) {
			data := getData(b)
			b.ReportAllocs()
			td := NewWithDecay(benchmarkCompression, benchmarkDecayValue, benchmarkDecayEvery)
			td2 := NewWithDecay(benchmarkCompression, benchmarkDecayValue, benchmarkDecayEvery)
			for i := 0; i < b.N; i++ {
				td2.Add(data[i], 1)
				if int64(i)%bm.size == 0 && i != 0 {
					td.AddCentroidList(td2.Centroids())
					td2.Clear()
				}
			}
			if td2.Centroids().Len() > 0 {
				td.AddCentroidList(td2.Centroids())
			}
		})
		b.Run(bm.name+"-regular", func(b *testing.B) {
			data := getData(b)
			b.ReportAllocs()
			td := NewWithDecay(benchmarkCompression, benchmarkDecayValue, benchmarkDecayEvery)
			for i := 0; i < b.N; i++ {
				td.Add(data[i], 1)
			}
		})
	}
}

func getData(b *testing.B) []float64 {
	data := make([]float64, b.N)
	for i := 0; i < b.N; i++ {
		data[i] = getVal()
	}
	b.ResetTimer()
	return data
}

var ns = []int{1000, 1500, 2000, 5000, 10000, 100000, 1000000}

func BenchmarkHistoSerde(b *testing.B) {
	for _, bm := range ns {
		bm := bm
		b.Run(strconv.Itoa(bm), func(b *testing.B) {
			data := getData(b)
			b.ResetTimer()
			td := NewWithDecay(benchmarkCompression, benchmarkDecayValue, benchmarkDecayEvery)
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if i > 0 && i%bm == 0 {
					bb, err := td.MarshalBinary()
					if err != nil {
						b.Fail()
					}
					td.Clear()
					err = td.UnmarshalBinary(bb)
					if err != nil {
						b.Fail()
					}
					td.Clear()
				}
				td.Add(data[i], 1)
			}
		})
	}
}

func TestWriteOutFiles(t *testing.T) {
	if err := os.RemoveAll("output"); err != nil {
		t.Fatal(err.Error())
	}
	if err := os.Mkdir("output", 0755); err != nil {
		t.Fatal(err.Error())
	}
	//qs := []float64{0.01}
	//qs := []float64{0.01, 0.001, 1.0e-04, 1.0e-05, 1.0e-06}
	qs := []float64{0.1, 0.01, 0.001, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 1.0e-04, 1.0e-05, 1.0e-06}

	for _, test := range benchmarks {
		for _, q := range qs {
			f, err := os.OpenFile(fmt.Sprintf("output/%s_%f.csv", strings.ToUpper(test.name), q), os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
			if err != nil {
				t.Fatal(err.Error())
			}
			if _, err := f.WriteString("error_q,norm_error_q" + "\n"); err != nil {
				t.Fatal(err.Error())
			}
			f.Close()
		}

		f, err := os.OpenFile(fmt.Sprintf("output/%s_centroid_counts.csv", strings.ToUpper(test.name)), os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
		if err != nil {
			t.Fatal(err.Error())
		}
		if _, err := f.WriteString("centroid_count" + "\n"); err != nil {
			t.Fatal(err.Error())
		}

		f.Close()
	}
	data := make([]float64, 1000000)
	n := 1
	for i := 0; i < n; i++ {
		uniform := rand.New(rand.NewSource(uint64(time.Now().UnixNano())))
		for j := range data {
			data[j] = uniform.Float64() * 100
		}
		sorted := getSortedCopy(data)

		for _, test := range benchmarks {
			test := test
			t.Run(test.name, func(t *testing.T) {
				td := NewWithScaler(test.scale, 100, 0, 0)
				for _, v := range data {
					td.Add(v, 1)
				}
				for _, v := range td.Centroids() {
					fmt.Println(v.Mean, v.Weight)
				}
				{
					f, err := os.OpenFile(fmt.Sprintf("output/%s_centroid_counts.csv", strings.ToUpper(test.name)), os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
					if err != nil {
						t.Fatal(err.Error())
					}
					if _, err := f.WriteString(fmt.Sprintf("%d\n", len(td.Centroids()))); err != nil {
						t.Fatal(err.Error())
					}
					f.Close()
				}
				for _, q := range qs {
					x1 := quantileOnSorted(q, sorted)
					q1 := cdfOnSorted(x1, sorted)
					q2 := td.CDF(x1)
					fmt.Println(x1, q1, q2, math.Abs(q1-q2), td.Min(), sorted[0])
					errq := math.Abs(q1 - q2)
					normerr := errq / math.Min(q, 1-q)
					f, err := os.OpenFile(fmt.Sprintf("output/%s_%f.csv", strings.ToUpper(test.name), q), os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
					if err != nil {
						t.Fatal(err.Error())
					}
					if _, err := f.WriteString(fmt.Sprintf("%f,%f\n", errq, normerr)); err != nil {
						t.Fatal(err.Error())
					}
					f.Close()
				}
			})
		}
	}
}
