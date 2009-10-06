#ifndef CONVERTXY_PYPLOTTING_HPP
#define CONVERTXY_PYPLOTTING_HPP

#include <iostream>
#include <sstream>
#include <map>
#include <string>

#include <cvd/image.h>
#include <cvd/image_io.h>

#include "Python.h"
#include <numpy/arrayobject.h>

#include "PyCVD.hpp"
#include "PyTooN.hpp"
#include "PySTL.hpp"
//#include <CXX/WrapPython.h>
//#include <CXX/Objects.hxx>


namespace Tesla {

  using namespace CVD;
  using namespace std;
  using namespace TooN;
  using namespace PyCPP;

  namespace PyInterface {
    static void checkPyError(const string &ran = "") {
      if (PyErr_Occurred()) {
	if (ran == "") {
	  cerr << "Python exception caught in C++ PyInterface!" << endl;
	}
	else {
	  cerr << "Python exception caught in C++ PyInterface when running " << ran << endl;
	}
	PyErr_Print();
      }
    }
  
    static PyObject *getScope() {
      static PyObject *scope = 0;
      if (scope == 0) {
	PyObject *__main__ = PyImport_AddModule("__main__");
	scope = PyObject_GetAttrString(__main__, "__dict__");
      }
      return scope;
    }

    static void run(const string &s) {
      PyRun_String(s.c_str(), Py_file_input, getScope(), getScope());
      checkPyError(s);
    }

    static void setvar(const string &s, PyObject *o) {
      PyDict_SetItemString(getScope(), s.c_str(), o);
      checkPyError();
    }

    static PyObject* getvar(const string &s) {
      PyObject *obj = PyDict_GetItemString(getScope(), s.c_str());
      if (obj == 0) {
	throw std::string("Variable " + s + " not available");
      }
      checkPyError();
      return obj;
    }

    static void delvar(const string &s) {
      PyDict_DelItemString(getScope(), s.c_str());
      checkPyError();
    }
    
    static int initializePython() {
      Py_Initialize();
      import_array1(1);
      //Py::Dict foo;
      run("import matplotlib.pylab as mpl");
      run("import numpy as np");
      return 0;
    }

    static int initializeGGFE() {
      run("try:\n  import ggfe.image_features\nexcept ImportError:\n  print 'ggfe could not be imported!'");
      run("image_grammar = ggfe.image_features.get_image_grammar()");
      return 0;
    }

    template <class T>
    static void imshow(const BasicImage <T> &x, string cmap = "") {
      PyObject *_in = 0;
      convert(x, (PyObject*&)_in);
      setvar("_in", _in);
      ostringstream out;
      out << "mpl.imshow(_in";
      if (cmap != "") {
	out << ", cmap=mpl.cm." << cmap;
      }
      out << ")";
      run(out.str());
      delvar("_in");
    }


    template <int M, int N, class T, class B>
    static void imshow(const Matrix <M, N, T, B> &x, string cmap = "jets") {
      PyObject *_in = 0;
      convert(x, (PyArrayObject*&)_in);
      setvar("_in", _in);
      run("mpl.imshow(_in);");
      delvar("_in");
    }

    template <int N, class T, class B>
    static void plot(const Vector <N, T, B> &x, string comp = "") {
      PyObject *_x = 0;
      convert(x, (PyArrayObject*&)_x);
      setvar("_x", _x);
      ostringstream out;
      out << "mpl.plot(_x";
      if (comp != "") {
	out << ", '" << comp << "'";
      }
      out << ")";
      run(out.str());
      delvar("_x");
    }

    template <int NX, class TX, class BX, int NY, class TY, class BY>
    static void plot(const Vector <NX, TX, BX> &x,
		     const Vector <NY, TY, BY> &y, string comp = "") {
      PyObject *_x = 0;
      PyObject *_y = 0;
      convert(x, (PyArrayObject*&)_x);
      convert(y, (PyArrayObject*&)_y);
      setvar("_x", _x);
      setvar("_y", _y);
      ostringstream out;
      out << "mpl.plot(_x, _y";
      if (comp != "") {
	out << ", '" << comp << "'";
      }
      out << ")";
      run(out.str());
      delvar("_x");
    }

    static void title(const string &s) {
      ostringstream o;
      o << "mpl.title('" << s << "')";
      run(o.str());
    }

    static void xlabel(const string &s) {
      ostringstream o;
      o << "mpl.xlabel('" << s << "')";
      run(o.str());
    }

    static void ylabel(const string &s) {
      ostringstream o;
      o << "mpl.ylabel('" << s << "')";
      run(o.str());
    }

    static void colorbar() {
      run("mpl.colorbar()");
    }

    static void figure(int i = -1) {
      if (i == -1) {
	run("mpl.figure()");
      }
      else {
	ostringstream out;
	out << "mpl.figure(" << i << ")";
	run(out.str());
      }
    }

    static void show() {
      run("mpl.show()");
    }

    static void subplot(int i) {
      ostringstream out;
      out << "mpl.subplot(" << i << ")";
      run(out.str());
    }

    static void subplot(int m, int n, int i) {
      ostringstream out;
      out << "mpl.subplot(" << m << "," << n << "," << i << ")";
      run(out.str());
    }

    static void savefig(const string &fn, const string &format = "") {
      ostringstream out;
      out << "mpl.savefig('" << fn << "'";
      if (format != "") {
	out << ", format='" << format << "'";
      }
      out << ")";
      run(out.str());
    }

    static string ggfe_generate_random_feature() {
      string retval;
      run("fstr = str(image_grammar.Feature(ggfe.Variable('IMG')))");
      PyObject *fstr(0);
      fstr = getvar("fstr");
      convert(PyCPP::py_cast<PyStringObject*>(fstr), retval);
      //delvar("fstr");
      return retval;
    }

    template <class TR, class TI>
    static Image<TR> run_feature(const string &s, Image<TI> &in) {
      Image<TR> result;
      PyArrayObject *_in(0);
      PyStringObject *_s(0);
      convert(s, _s);
      convert(in, (PyObject*&)_in);
      setvar("IMG", (PyObject*)_in);
      setvar("PS", (PyObject*)_s);
      run("RESULT = np.asarray(ggfe.image_features.evaluate_feature(image_grammar, PS, IMG), dtype='d')");
      PyArrayObject *_out = PyCPP::py_cast<PyArrayObject*>(getvar("RESULT"));
      convert(_out, result);
      //imshow(result);
      //show();
      delvar("IMG");
      delvar("PS");
      return result;
    }

    static vector<ImageRef> parse_centroid_file(const string &fn) {
      run("fid = open('" + fn + "')");
      run("centroids = np.asarray(eval(fid.readlines()[0]), dtype='i')");
      vector <ImageRef> cents;
      convert((PyArrayObject*)getvar("centroids"), cents);
      run("fid.close()");
      run("del fid");
      run("del centroids");
      return cents;
    }
  }
  //  };
}

#endif
