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


namespace PyCPP {

  using namespace CVD;
  using namespace std;
  using namespace TooN;
  using namespace PyCPP;

  namespace Plotting {

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

    static bool hasvar(const string &s) {
      PyStringObject *_key(0);
      convert(s, _key);
      int result = PyDict_Contains(getScope(), (PyObject*)_key);
      Py_XDECREF(_key);
      checkPyError();
      return (bool)result;
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
      run("from numpy import inf, nan");
      return 0;
    }

    static int initializeGGFE() {
      run("try:\n  import ggfe.image_features\nexcept ImportError:\n  print 'ggfe could not be imported!'");
      run("try:\n  import ggfe.approximate\nexcept ImportError:\n  print 'ggfe kernel approximators cannot be imported!'");
      run("try:\n  import ggfe.bioid\nexcept ImportError:\n  print 'ggfe BioID reader cannot be imported!'");
      run("grammars = {}");
      run("grammars['main'] = ggfe.image_features.get_image_grammar()");
      run("grammars['haar'] = ggfe.image_features.get_haar_grammar()");
      run("grammars['haar-str'] = ggfe.image_features.get_haar_string_grammar()");
      return 0;
    }

    static int initializeGGFEDiagnostics() {
      run("try:\n  import ggfe.random_hypotheses\nexcept ImportError:\n  print 'ggfe.random_hypotheses could not be imported!'");
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


    static void plot(const vector <ImageRef> &pts, string comp = "") {
      PyObject *_pts = 0;
      convert(pts, (PyArrayObject*&)_pts);
      setvar("_pts", _pts);
      ostringstream out;
      out << "mpl.plot(_pts[:,1], _pts[:,0] ";
      if (comp != "") {
	out << ", '" << comp << "'";
      }
      out << ")";
      run(out.str());
      delvar("_pts");
    }

    static void plot(const set <ImageRef> &pts, string comp = "") {
      PyObject *_pts = 0;
      convert(pts, (PyArrayObject*&)_pts);
      setvar("_pts", _pts);
      ostringstream out;
      out << "mpl.plot(_pts[:,1], _pts[:,0] ";
      if (comp != "") {
	out << ", '" << comp << "'";
      }
      out << ")";
      run(out.str());
      delvar("_pts");
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

    static void clf() {
      run("mpl.clf()");
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

    template <class P>
    static void get_conic_kernel(int r, Matrix<-1,-1, P> &out) {
      ostringstream ostr;
      ostr << "KERNEL = np.asarray(ggfe.approximate.create_cache(" << r
	   << ", 'conic'), dtype=np.float)" << endl;
      run(ostr.str());
      PyArrayObject *_out = PyCPP::py_cast<PyArrayObject*>(getvar("KERNEL"));
      convert(_out, out);
      delvar("KERNEL");
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

    static string ggfe_generate_random_feature(const string &grammar="main") {
      string retval;
      ostringstream ostr;
      ostr << "fstr = str(grammars['"
	   << grammar
	   << "'].Feature(ggfe.Variable('IMG')))";
      run(ostr.str());
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
      ostringstream ostr;
      ostr << "RESULT = np.asarray(ggfe.image_features.evaluate_feature("
	   << "PS, IMG), dtype='d')" << endl;
      run(ostr.str());
      PyArrayObject *_out = PyCPP::py_cast<PyArrayObject*>(getvar("RESULT"));
      convert(_out, result);
      //imshow(result);
      //show();
      delvar("IMG");
      delvar("PS");
      delvar("RESULT");
      Py_XDECREF(_in);
      Py_XDECREF(_s);
      //Py_XDECREF(_out);
      return result;
    }

    static vector<ImageRef> parse_centroid_file(const string &fn) {
      run("fid = open('" + fn + "')");
      run("centroids = np.asarray(eval(fid.readlines()[0]), dtype='i')");
      run("centroids = centroids[:,::-1].copy()");
      vector <ImageRef> cents;
      convert((PyArrayObject*)getvar("centroids"), cents);
      /**for (size_t i = 0; i < cents.size(); i++) {
	int temp = cents[i].x;
	cents[i].x = cents[i].y;
	cents[i].y = temp;
	}**/
      run("fid.close()");
      run("del fid");
      run("del centroids");
      return cents;
    }

    static vector<ImageRef> parse_bioid_file(const string &fn, const string &kw) {
      ostringstream ostr;
      ostr << "centroids = np.array(ggfe.bioid.get_points_by_keyword('"
	   << fn << "', '" << kw << "'), dtype='i')" << endl;
      ostr << "centroids = centroids[:,::-1].copy()" << endl;
      run(ostr.str());
      vector <ImageRef> cents;
      convert((PyArrayObject*)getvar("centroids"), cents);
      run("del centroids");
      return cents;
    }

    template <class TW>
    static void artificial_hits(const ImageRef &sz, const set<ImageRef> &centers, double hit_spread_max,
				double dr, double far, int hits_per_detection,
				map<TW, vector<ImageRef>, std::greater<TW> > &hit_map) {
      ostringstream ostr;
      PyArrayObject *_centers(0);
      convert(centers, _centers);
      setvar("centers", (PyObject*)_centers);
      ostr << "(hits, confidences) = ggfe.random_hypotheses.artificial_hits("
	   << "[" << sz.y << ", " << sz.x << "], "
	   << "centers, "
	   << "hit_spread_max=" << hit_spread_max << ", "
	   << "dr=" << dr << ", "
	   << "far=" << far << ", "
	   << "hits_per_detection=" << hits_per_detection << ")";

      run(ostr.str());
      vector<ImageRef> hits;

      PyArrayObject *_hits(0);
      PyArrayObject *_confidences(0);

      _hits = (PyArrayObject*)getvar("hits");
      _confidences = (PyArrayObject*)getvar("confidences");

      convert(_hits, hits);

      Vector<-1, double> confidences(hits.size());

      convert(_confidences, confidences);

      //map<TW, vector<ImageRef>, std::greater<TW> > hit_map;

      for (int i = 0; i < confidences.size(); i++) {
	hit_map[confidences[i]].push_back(hits[i]);
      }

      delvar("centers");
      delvar("hits");
      delvar("confidences");
    }
  }
  //  };
}

#endif
