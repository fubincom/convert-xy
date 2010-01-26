#ifndef CONVERTXY_TOON_HPP
#define CONVERTXY_TOON_HPP

#include <TooN/TooN.h>
#include <vector>
#include <sstream>

namespace ConvertXY {

  using namespace TooN;
  using namespace std;


  ///////////////////////////////////////////////////////////
  ////
  //// By default, an object factory for an arbitrary TooN
  //// type cannot be created and if explicitly instantiated
  //// will result in an incomplete type at compile time.
  ////
  ///////////////////////////////////////////////////////////
  template <int Rows, int Cols, class ElemType, class Layout>
  class CannotCreateTooNMatrixObjectFactory;

  template <int Rows, int Cols, class ElemType, class Layout>
  struct ObjectFactory<Matrix<Rows, Cols, ElemType, Layout> > {

    CannotCreateTooNMatrixObjectFactory<Rows, Cols, ElemType, Layout> no_factory_defined_for_layout;

  };

  template <int NElems, class ElemType, class Layout>
  class CannotCreateTooNVectorObjectFactory;

  template <int NElems, class ElemType, class Layout>
  struct ObjectFactory<Vector<NElems, ElemType, Layout> > {

    CannotCreateTooNVectorObjectFactory<NElems, ElemType, Layout> no_factory_defined_for_layout;

  };

  //////////////////////////////////////////////////////////
  ////
  //// Converters for Reference matrices.
  ////
  //////////////////////////////////////////////////////////
  template <int Rows, int Cols, class ElemType>
  class ObjectFactory<Matrix<Rows, Cols, ElemType, Reference::RowMajor> > {

  public:
    static Matrix<Rows, Cols, ElemType, Reference::RowMajor> create(const vector <size_t> &dimensions, void *bufferPtr = 0) {
      check(dimensions);
      return Matrix<Rows, Cols, ElemType, Reference::RowMajor>((ElemType*)bufferPtr, dimensions[0], dimensions[1]);
    }


    static Matrix<Rows, Cols, ElemType, Reference::RowMajor>* create_ptr(const vector <size_t> &dimensions, void *bufferPtr = 0) {
      check(dimensions);
      return new Matrix<Rows, Cols, ElemType, Reference::RowMajor>((ElemType*)bufferPtr, dimensions[0], dimensions[1]);
    }

  private:

    static void check(const vector <size_t> &dimensions) {
      if (dimensions.size() != 2) {
	ostringstream ostr;
	ostr << "ObjectFactory<TooN::Matrix>: dimensions mismatch, " << dimensions.size() << " expected (2)";
	throw Py::TypeError(ostr.str());
      }
      if (Rows != -1 && dimensions[0] != Rows) {
	ostringstream ostr;
	ostr << "ObjectFactory<TooN::Matrix>: row mismatch, " << dimensions[0] << " expected (" << Rows << ")";
	throw Py::TypeError(ostr.str());
      }
      if (Cols != -1 && dimensions[1] != Cols) {
	ostringstream ostr;
	ostr << "ObjectFactory<TooN::Matrix>: column mismatch, " << dimensions[1] << " expected (" << Cols << ")";
	throw Py::TypeError(ostr.str());
      }      
    }
  };

  template <int Rows, int Cols, class ElemType, class BA>
  struct DefaultToCPPConvertAction<Matrix<Rows, Cols, ElemType, Reference::RowMajor>, BA > {
    typedef Reuse<> Action;
  };

  template <int Rows, int Cols, class ElemType, class Layout, class BA>
  struct DefaultToCPPConvertAction<Matrix<Rows, Cols, ElemType, Layout>, BA > {
    typedef Copy<> Action;
  };

  template <>
  template <int Rows, int Cols, class ElemType>
  struct ConvertToCPP<Matrix<Rows, Cols, ElemType, Reference::RowMajor>, PyArrayObject*, Reuse<> >
    : public ConvertToCPPBase<Matrix<Rows, Cols, ElemType, Reference::RowMajor>, Reuse<> > {

    ConvertToCPP() {}
    virtual ~ConvertToCPP() {}

    void convert(PyObject *src, Matrix<Rows, Cols, ElemType, Reference::RowMajor> &dst) const {
      if (PyArray_TYPE(src) != NumPyType<ElemType>::numpy_code) {
	throw Py::RuntimeError("ConvertXY: type unsupported for reuse conversion to BasicImage.");
      }
      vector <size_t> sz(getSize(src));
      if (sz.size() != 2) {
	throw Py::RuntimeError("ConvertXY: array must be two dimensional for reuse conversion to TooN::Matrix.");
      }
      if (PyArray_ISCONTIGUOUS(src)) {
	/** In the future, this check needs to be enabled when the .data()
            function is provided by Vector and Matrix to ensure the
            ObjectFactory was invoked correctly. */
	//if ((T*)PyArray_DATA(src) != dst.data()) {
	///  throw Py::RuntimeError("ConvertXY: factory invoked incorrectly.");
	//}
	return;
      }
      else {
	throw Py::RuntimeError("ConvertXY: array must be contiguous for reuse conversion to TooN::Matrix.");
      }
    }

    virtual bool isImplemented() const { return true; }

    void *getBufferPointer(PyObject *obj) const {
      return PyArray_DATA(obj);
    }

    virtual vector <size_t> getSize(PyObject *src) const {
      vector <size_t> sz;
      Py::Sequence seq(src);
      const size_t ndims = PyArray_NDIM(src);
      if (ndims != 2) {
	throw Py::RuntimeError("only exactly two dimensional arrays may be converted to TooN matrices.");
      }
      for (size_t i = 0; i < ndims; i++) {
	sz.push_back(PyArray_DIM(src, i));
      }
      return sz;
    }
  };

  template <>
  template <int Rows, int Cols, class ElemType, class Layout>
  struct ConvertToCPP<Matrix<Rows, Cols, ElemType, Layout>, PyArrayObject*, Copy<> >
    : public ConvertToCPPBase<Matrix<Rows, Cols, ElemType, Layout>, Copy<> > {

    ConvertToCPP() {}
    virtual ~ConvertToCPP() {}

    void convert_object_array(PyObject *src, Matrix<Rows, Cols, ElemType, Layout> &dst) const {
      if (PyArray_NDIM(src) == 2) {
	int ysize = PyArray_DIM(src, 0);
	int xsize = PyArray_DIM(src, 1);
	if (dst.num_rows() != ysize || dst.num_cols() != xsize) {
	  throw Py::RuntimeError("ConvertXY: ToCPP<Matrix, Py::Sequence> size mismatch!");
	}
	PyArrayIterObject* iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)src);
	if (iter == 0) {
	  throw Py::RuntimeError("ConvertXY: ToCPP<Matrix, Py::Sequence> Could not grab iterator!");
	}

	for (int y = 0; y < ysize; y++) {
	  for (int x = 0; iter->index < iter->size && x < xsize; x++) {
	    PyObject *item = (PyObject*)iter->dataptr;
	    Py::Object _item(item);
	    ConvertToCPPBase<ElemType, Copy<> >
	      &converter(ToCPPDispatch<ElemType, Copy<> >::getConverter(item));
	    converter.convert(item, dst(y, x));
	    PyArray_ITER_NEXT(iter);
	  }
	}    
      }
      else {
	throw Py::RuntimeError("ConvertXY: ToCPP<Matrix, Py::Sequence>");
      }
    }

    template <class WalkableTypeList>
    void walk_type_list(PyObject *src, Matrix<Rows, Cols, ElemType, Layout> &dst) const {
      typedef typename WalkableTypeList::Head HeadType;
      if (PyArray_TYPE(src) == NumPyType<HeadType>::numpy_code) {
	if (PyArray_NDIM(src) == 2) {
	  int ysize = PyArray_DIM(src, 0);
	  int xsize = PyArray_DIM(src, 1);
	  if (dst.num_rows() != ysize || dst.num_cols() != xsize) {
	    throw Py::RuntimeError("ConvertXY: ToCPP<Matrix, Py::Sequence> size mismatch!");
	  }
	  PyArrayIterObject* iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)src);
	  if (iter == 0) {
	    throw std::string("ConvertXY: ToCPP<Image<T>, NumPyArray> Could not grab iterator!");
	  }

	  for (int y = 0; y < ysize; y++) {
	    for (int x = 0; iter->index < iter->size && x < xsize; x++) {
	      HeadType *item = (HeadType*)iter->dataptr;
	      dst(y, x) = (ElemType)*item;
	      PyArray_ITER_NEXT(iter);
	    }
	  }
	}
	else {
	  throw Py::RuntimeError("ConvertXY: ToCPP<Matrix<Rows, Cols, NumPyArray>: NumPy array must be two-dimensional.");
	}
      }
      else {
	typedef typename WalkableTypeList::Tail TailType;
	if (IsEnd<TailType>::end) {
	  throw Py::TypeError("ConvertXY: No way to convert type");
	}
	else {
	  // to make the static type-checker happy.
	  walk_type_list<typename AdvanceIfPossible<WalkableTypeList>::choice >(src, dst);
	}
      }
    }
    
    virtual void convert(PyObject *src, Matrix<Rows, Cols, ElemType, Layout> &dst) const {
      Py::Object _src(src);
      if (PyArray_TYPE(src) == NPY_OBJECT || PyArray_TYPE(src) == NPY_STRING) {
	convert_object_array(src, dst);
      }
      else if (PyArray_TYPE(src) < NPY_OBJECT) {
	walk_type_list<NumPyFloatTypes>(src, dst);
      }
      else {
	throw Py::TypeError("NumPy type code not understood!");
      }
    }

    virtual bool isImplemented() const { return true; }

    void *getBufferPointer(PyObject *obj) const {
      return PyArray_DATA(obj);
    }

    virtual vector <size_t> getSize(PyObject *src) const {
      vector <size_t> sz;
      Py::Sequence seq(src);
      const size_t ndims = PyArray_NDIM(src);
      for (size_t i = 0; i < ndims; i++) {
	sz.push_back(PyArray_DIM(src, i));
      }
      return sz;
    }

  };

  template <int Rows, int Cols, class ElemType>
  class ObjectFactory<Matrix<Rows, Cols, ElemType, RowMajor> > {

  public:
    static Matrix<Rows, Cols, ElemType, RowMajor> create(const vector <size_t> &dimensions, void *bufferPtr = 0) {
      check(dimensions);
      return Matrix<Rows, Cols, ElemType, RowMajor>((ElemType*)bufferPtr, dimensions[0], dimensions[1]);
    }

    static Matrix<Rows, Cols, ElemType, RowMajor>* create_ptr(const vector <size_t> &dimensions, void *bufferPtr = 0) {
      check(dimensions);
      return new Matrix<Rows, Cols, ElemType, RowMajor>((ElemType*)bufferPtr, dimensions[0], dimensions[1]);
    }

  private:

    static void check(const vector <size_t> &dimensions) {
      if (dimensions.size() != 2) {
	ostringstream ostr;
	ostr << "ObjectFactory<TooN::Matrix>: number of dimensions mismatch, " << dimensions.size() << " expected (2)";
	throw Py::TypeError(ostr.str());
      }
      if (Rows != -1 && dimensions[0] != Rows) {
	ostringstream ostr;
	ostr << "ObjectFactory<TooN::Matrix>: row mismatch, " << dimensions[0] << " expected (" << Rows << ")";
	throw Py::TypeError(ostr.str());
      }
      if (Cols != -1 && dimensions[1] != Cols) {
	ostringstream ostr;
	ostr << "ObjectFactory<TooN::Matrix>: column mismatch, " << dimensions[1] << " expected (" << Cols << ")";
	throw Py::TypeError(ostr.str());
      }
    }
  };

  template <int Rows, int Cols, class ElemType, class Layout, class DefaultBufferAction>
  struct DefaultToPythonStructure<Matrix<Rows, Cols, ElemType, Layout>, DefaultBufferAction > {
    typedef DefaultToPythonStructure<ElemType, DefaultBufferAction> ElemClass;
    typedef typename ElemClass::Structure ElemStructure;
    typedef PyArray<ElemStructure, Copy<> > Structure;
  };


  //////////////////////////////////////////////////////////
  ////
  //// Converters for non-reference matrices (i.e. matrices
  //// that own their own memory).
  ////
  //////////////////////////////////////////////////////////


  //////////////////////////////////////////////////////////
  ////
  //// Converters for Reference vectors.
  ////
  //////////////////////////////////////////////////////////
  
  template <int NElems, class ElemType, class Layout, class BA>
  struct DefaultToPythonStructure<Vector<NElems, ElemType, Layout>, BA > {
    typedef DefaultToPythonStructure<ElemType, BA> ElemClass;
    typedef typename ElemClass::Structure ElemStructure;
    typedef PyArray<ElemStructure, Copy<> > Structure;
  };

  template <int NElems, class ElemType, class BA>
  struct DefaultToCPPConvertAction<Vector<NElems, ElemType, Reference>, BA > {
    typedef Reuse<> Action;
  };


  //////////////////////////////////////////////////////////
  ////
  //// Converters for non-reference vectors (i.e. vectors
  //// that own their own memory).
  ////
  //////////////////////////////////////////////////////////

  template <int NElems, class ElemType, class Layout, class BA>
  struct DefaultToCPPConvertAction<Vector<NElems, ElemType, Layout>, BA > {
    typedef Copy<> Action;
  };

}

#endif