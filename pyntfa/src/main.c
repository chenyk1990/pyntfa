#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <numpy/arrayobject.h>
#include "ntfa_divnnsc.h"

#define SF_MAX(a,b) ((a) < (b) ? (b) : (a))
#define SF_MIN(a,b) ((a) < (b) ? (a) : (b))
#define SF_MAX_DIM 9
#define SF_PI (3.14159265358979323846264338328)

#include "ntfa_alloc.h"

int kiss_fft_next_fast_size(int n)
{
    while(1) {
        int m=n;
        while ( (m%2) == 0 ) m/=2;
        while ( (m%3) == 0 ) m/=3;
        while ( (m%5) == 0 ) m/=5;
        if (m<=1)
            break; /* n is completely factorable by twos, threes, and fives */
        n++;
    }
    return n;
}

/*from user/chenyk/vecoper.c */
static PyObject *ntft(PyObject *self, PyObject *args){
    
	/**initialize data input**/
    int nd, nd2;
    
    PyObject *f1=NULL;
    PyObject *arrf1=NULL;

	int ndata;	/*integer parameter*/
	float fpar; /*float parameter*/
    int ndim, i;
    float *data;
    
	PyArg_ParseTuple(args, "Oif", &f1, &ndata, &fpar);

    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
    
    nd2=PyArray_NDIM(arrf1);
    
    npy_intp *sp=PyArray_SHAPE(arrf1);

	data  = (float*)malloc(ndata * sizeof(float));
	
    if (*sp != ndata)
    {
    	printf("Dimension mismatch, N_input = %d, N_data = %d\n", *sp, ndata);
    	return NULL;
    }
    
    /*reading data*/
    for (i=0; i<ndata; i++)
    {
        data[i]=*((float*)PyArray_GETPTR1(arrf1,i));
    }

	/*sub-function goes here*/
	
	
    /*Below is the output part*/
    PyArrayObject *vecout;
	npy_intp dims[2];
	dims[0]=ndata;dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	for(i=0;i<dims[0];i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = data[i];
	
	return PyArray_Return(vecout);
	
}


static PyObject *tf1d(PyObject *self, PyObject *args){
    
	/**initialize data input**/
    int nd, nd2;
    
    PyObject *f1=NULL;
    PyObject *arrf1=NULL;

	int ndata;	/*integer parameter*/
	float fpar; /*float parameter*/
    int ndim, i;
    float *data;
    
    int niter,verb,rect0,n1;
    float dt,alpha;
    int ifb,inv;
	PyArg_ParseTuple(args, "Oiiiiiiff", &f1,&niter,&n1,&verb,&rect0,&ifb,&inv,&dt,&alpha);
	
	ndata=n1;


    int i1, iw, nt, nw, i2, n2, n12, n1w;
    int m[SF_MAX_DIM], *rect;
    float t, d1, w, w0, dw, mean=0.0f;
    float *trace, *kbsc, *mkbsc, *sscc, *mm, *ww;
    
    d1=dt;
//     if (!inv) {
// 	if (!tf_getint("nw",&nw)) { /* number of frequencies */
	    nt = 2*kiss_fft_next_fast_size((n1+1)/2);
	    nw = nt/2+1;
	    dw = 1./(nt*d1);
	    w0 = 0.;
// 	} else {
// 	    if (!tf_getfloat("dw",&dw)) {
// 		/* frequency step */
// 		nt = 2*kiss_fft_next_fast_size((n1+1)/2);
// 		dw = 1./(nt*d1);
// 	    }
// 	    if (!tf_getfloat("w0",&w0)) w0=0.;
// 	    /* first frequency */
// 	}
// 	n2 = tf_leftsize(in,1);
// 	tf_shiftdim(in, out, 2);
// 	tf_putint(out,"n2",nw);
// 	tf_putfloat(out,"d2",dw);
// 	tf_putfloat(out,"o2",w0);
// 	tf_putstring(out,"label2","Frequency");
// 	tf_putstring(out,"unit2","Hz");
// 	tf_settype(out,tf_COMPLEX);

// 	if (!tf_getint("rect",&rect0)) rect0=10;
// 	/* smoothing radius (in time, samples) */
// 	if (!tf_getint("niter",&niter)) niter=100;
// 	/* number of inversion iterations */
// 	if (!tf_getfloat("alpha",&alpha)) alpha=0.;
	/* frequency adaptivity */

	for(i2=0; i2 < SF_MAX_DIM; i2 ++) {
	    m[i2] = 1;
	}
	m[0] = n1;
//     } else {
// 	n2 = tf_leftsize(in,2);
// 	if (!tf_histint(in,"n2",&nw)) tf_error("No n2= in input");
// 	if (!tf_histfloat(in,"d2",&dw)) tf_error("No d2= in input");
// 	if (!tf_histfloat(in,"o2",&w0)) tf_error("No o2= in input");
// 	tf_unshiftdim(in, out, 2);
// 	tf_settype(out,tf_FLOAT);
//     }

//     if (NULL != basis) {
// 	tf_shiftdim(in, basis, 2);
// 	tf_putint(basis,"n2",nw);
// 	tf_putfloat(basis,"d2",dw);
// 	tf_putfloat(basis,"o2",w0);
// 	tf_putstring(basis,"label2","Frequency");
// 	tf_putstring(basis,"unit2","Hz");
//     }

    n1w = n1*nw;
    n12 = 2*n1w;
    dw *= 2.*SF_PI;
    w0 *= 2.*SF_PI;


	printf("niter=%d,n1=%d,nw=%d,nt=%d,n12=%d,rect0=%d\n",niter,n1,nw,nt,n12,rect0);
	printf("dw=%g,w0=%g,alpha=%g,dt=%g\n",dw,w0,alpha,dt);

	

    trace = tf_floatalloc(n1);
    kbsc    = tf_floatalloc(n12); /*kbsc is the basis functions*/
//     outp = tf_complexalloc(n1w);
//     cbsc = tf_complexalloc(n1w);

    rect = tf_intalloc(2*nw);
    for (iw=0; iw < nw; iw++) {
	rect[iw+nw] = rect[iw] = SF_MAX(1, (int) rect0/(1.0+alpha*iw/nw));
    }

//     if (!inv) {
	sscc = tf_floatalloc(n12);
	divnn_sc_init(2*nw, 1, n1, m, rect, kbsc, 
			(bool) (verb && (n2 < 500))); 
//     } else {
// 	sscc = NULL;
//     }
    
//     if (NULL != tf_getstring("mask")) { /* data weight */
// 	mask = tf_input("mask");
// 	mm = tf_floatalloc(n1);	
//     } else {
// 	mask = NULL;
// 	mm = NULL;
//     }
// 	mask = NULL;
	mm = NULL;
//     if (NULL != tf_getstring("weight")) { /* model weight */
// 	weight = tf_input("weight");
// 	ww = tf_floatalloc(n1w);
//     } else {
// 	weight = NULL;
// 	ww = NULL;
//     }
// 	weight = NULL;
	ww = NULL;

    /* sin and cos basis */
    for (iw=0; iw < nw; iw++) {
        w = w0 + iw*dw;
	for (i1=0; i1 < n1; i1++) {
	    if (0.==w) { /* zero frequency */
		kbsc[iw*n1+i1] = 0.;
	    } else {
		t = i1*d1;
		kbsc[iw*n1+i1] = sinf(w*t);
	    }
	}
    }
    for (iw=0; iw < nw; iw++) {
        w = w0 + iw*dw;
	for (i1=0; i1 < n1; i1++) {
	    if (0.==w) { /* zero frequency */
		kbsc[(iw+nw)*n1+i1] = 0.5;
	    } else {
		t = i1*d1;
		kbsc[(iw+nw)*n1+i1] = cosf(w*t);
	    }

// 	    cbsc[iw*n1+i1] = tf_cmplx(kbsc[(iw+nw)*n1+i1],
// 				      kbsc[iw*n1+i1]);
	}
    }

    
    if (NULL != mm || NULL != ww) {
	mkbsc = tf_floatalloc(n12);
	for (i1=0; i1 < n12; i1++) {
	    mkbsc[i1] = kbsc[i1];
	}
    } else {
	mkbsc = NULL;

	mean = 0.;
	for (i1=0; i1 < n12; i1++) {
	    mean += kbsc[i1]*kbsc[i1];
	}
	mean = sqrtf (mean/(n12));
	for (i1=0; i1 < n12; i1++) {
	    kbsc[i1] /= mean;
	}
    }
    


// 	if (NULL != basis) tf_complexwrite(cbsc,n1w,basis);

	if (NULL != mm || NULL != ww) {
	    for (i1=0; i1 < n12; i1++) {
		kbsc[i1] = mkbsc[i1];
	    }

	    if (NULL != mm) {
// 		tf_floatread(mm,n1,mask);
		for (iw=0; iw < 2*nw; iw++) {
		    for (i1=0; i1 < n1; i1++) {
			kbsc[iw*n1+i1] *= mm[i1];
		    }
		}
	    }

	    if (NULL != ww) {
// 		tf_floatread(ww,n1w,weight);
		for (iw=0; iw < nw; iw++) {
		    for (i1=0; i1 < n1; i1++) {
			kbsc[iw*n1+i1]      *= ww[iw*n1+i1];
			kbsc[(iw+nw)*n1+i1] *= ww[iw*n1+i1];
		    }
		}
	    }

	    mean = 0.;
	    for (i1=0; i1 < n12; i1++) {
		mean += kbsc[i1]*kbsc[i1];
	    }
	    mean = sqrtf (mean/(n12));
	    for (i1=0; i1 < n12; i1++) {
		kbsc[i1] /= mean;
	    }
	}

// 	if (!inv) {
// 	    tf_floatread(trace,n1,in);
	    
	if(!inv)
	{
    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
    
    nd2=PyArray_NDIM(arrf1);
    
    npy_intp *sp=PyArray_SHAPE(arrf1);

	data  = (float*)malloc(ndata * sizeof(float));
	
    if (*sp != ndata)
    {
    	printf("Dimension mismatch, N_input = %d, N_data = %d\n", *sp, ndata);
    	return NULL;
    }
    
    /*reading data*/
    for (i=0; i<ndata; i++)
    {
        trace[i]=*((float*)PyArray_GETPTR1(arrf1,i));
    }
	printf("ndata=%d\n",ndata);
	    
	    if (NULL != mm) {
		for (i1=0; i1 < n1; i1++) {
		    trace[i1] *= mm[i1];
		}
	    }
	    
	    for(i1=0; i1 < n1; i1++) {
		trace[i1] /= mean;
	    }
	    divnn_sc (trace,sscc,niter); /*YC 10/20/2024: sscc is the NTF coefficients*/
	}else{
	/*This part is to reconstruct the data given the basis functions and their weights (i.e., TF spectrum)*/
	
    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
    
    nd2=PyArray_NDIM(arrf1);
    
    npy_intp *sp=PyArray_SHAPE(arrf1);
    	
    for (i=0; i<n1w*2; i++)
    {
        sscc[i]=*((float*)PyArray_GETPTR1(arrf1,i));
    }

	    for (i1=0; i1 < n1; i1++) {
		trace[i1] = 0.;
	    }
	    
	    for (iw=0; iw < nw; iw++) {
		for (i1=0; i1 < n1; i1++) {
		    trace[i1] += sscc[(iw+nw)*n1+i1]*kbsc[(iw+nw)*n1+i1]
			*mean+sscc[iw*n1+i1]*kbsc[iw*n1+i1]*mean;
		    if (NULL != mm) trace[i1] *= mm[i1];
		}
	    }
	}
// 	    for (iw=0; iw < nw; iw++) {
// 		for (i1=0; i1 < n1; i1++) {
// 		    outp[iw*n1+i1] = tf_cmplx(sscc[(iw+nw)*n1+i1],
// 					      sscc[iw*n1+i1]);
// 		}
// 	    }

// 	    if (NULL != ww) {
// 		for (i1=0; i1 < n1w; i1++) {
// #ifdef tf_HAS_COMPLEX_H
// 		    outp[i1] *= ww[i1];
// #else
// 		    outp[i1] = tf_crmul(outp[i1],ww[i1]);
// #endif
// 		}
// 	    } 

// 	    tf_complexwrite(outp,n1w,out);
// 	} else {
// 	    for (i1=0; i1 < n1; i1++) {
// 		trace[i1] = 0.;
// 	    }
// 	    tf_complexread(outp,n1w,in);
// 	    for (iw=0; iw < nw; iw++) {
// 		for (i1=0; i1 < n1; i1++) {
// 		    trace[i1] += crealf(outp[iw*n1+i1])*kbsc[(iw+nw)*n1+i1]
// 			*mean+cimagf(outp[iw*n1+i1])*kbsc[iw*n1+i1]*mean;
// 		    if (NULL != mm) trace[i1] *= mm[i1];
// 		}
// 	    }
// 	    tf_floatwrite(trace,n1,out);
// 	}





	/*sub-function goes here*/
	
	
	
    /*Below is the output part*/
    PyArrayObject *vecout;
    npy_intp dims[2];
    
    if(!inv)
    {

	if(ifb) /*if output the basis functions, e.g., Fourier bases in this case*/
	{
	dims[0]=ndata*nw*4+3;dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	for(i=0;i<ndata*nw*2;i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = sscc[i];
	for(i=0;i<ndata*nw*2;i++)
		(*((float*)PyArray_GETPTR1(vecout,i+ndata*nw*2))) = kbsc[i];
		
	(*((float*)PyArray_GETPTR1(vecout,0+ndata*nw*4))) = w0;
	(*((float*)PyArray_GETPTR1(vecout,1+ndata*nw*4))) = dw;
	(*((float*)PyArray_GETPTR1(vecout,2+ndata*nw*4))) = nw;
	}else{
	dims[0]=ndata*nw*2+3;dims[1]=1;
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	for(i=0;i<ndata*nw*2;i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = sscc[i];
	printf("w0=%g,dw=%g,nw=%d\n",w0,dw,nw);
	(*((float*)PyArray_GETPTR1(vecout,0+ndata*nw*2))) = w0;
	(*((float*)PyArray_GETPTR1(vecout,1+ndata*nw*2))) = dw;
	(*((float*)PyArray_GETPTR1(vecout,2+ndata*nw*2))) = nw;
	}
	
	
	}else{
	
	dims[0]=n1;dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	for(i=0;i<dims[0];i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = trace[i];
	}
	
	return PyArray_Return(vecout);
	
}

/*documentation for each functions.*/
static char ntfacfun_document[] = "Document stuff for this C module...";

/*defining our functions like below:
  function_name, function, METH_VARARGS flag, function documents*/
static PyMethodDef functions[] = {
  {"ntft", ntft, METH_VARARGS, ntfacfun_document},
  {"tf1d", tf1d, METH_VARARGS, ntfacfun_document},
  {NULL, NULL, 0, NULL}
};

/*initializing our module informations and settings in this structure
for more informations, check head part of this file. there are some important links out there.*/
static struct PyModuleDef ntfacfunModule = {
  PyModuleDef_HEAD_INIT, /*head informations for Python C API. It is needed to be first member in this struct !!*/
  "ntfacfun",  /*module name*/
  NULL, /*means that the module does not support sub-interpreters, because it has global state.*/
  -1,
  functions  /*our functions list*/
};

/*runs while initializing and calls module creation function.*/
PyMODINIT_FUNC PyInit_ntfacfun(void){
  
    PyObject *module = PyModule_Create(&ntfacfunModule);
    import_array();
    return module;
}
