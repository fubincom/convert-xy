#include <iostream>

#include "converter.hpp"
#include "STL.hpp"
#include "holder.hpp"

#include <CXX/Objects.hxx>

using namespace std;
using namespace Py;
using ConvertXY::PyTuple;
using ConvertXY::PySet;
using ConvertXY::PyString;

void initialize() {
  Py_Initialize();
  //import_array();
}

int main() {
  initialize();
  List list;
  list.append(Long(1));
  list.append(Long(2));
  cout << list << endl;
  vector <int> v;
  ConvertXY::convert(list.ptr(), v);
  Long el(5);

  for (size_t i = 0; i < v.size(); i++) {
    cout  << v[i] << ",";
  }
  cout << endl;


  PyObject *list2(0);

  list2 = ConvertXY::convert(v);
  Py::List list3(list2);
  cout << list3 << endl;
 

  vector<string> string_vec;

  string_vec.push_back("foo");
  string_vec.push_back("bar");

  PyObject *string_list = ConvertXY::convert(string_vec);
  Py::List pstring_list(string_list);
  cout << pstring_list << endl;

  PyObject *string_tuple = ConvertXY::convert_override<PyTuple<PyString> >(string_vec);
  Py::Tuple pstring_tuple(string_tuple);
  cout << pstring_tuple << endl;

  PyObject *string_set = ConvertXY::convert_override<PySet<PyString> >(string_vec);
  Py::Object pstring_set(string_set);
  cout << pstring_set << endl;

  vector <string> get_set_back_as_vec;

  ConvertXY::convert(string_set, get_set_back_as_vec);
  for (size_t i = 0; i < v.size(); i++) {
    cout << get_set_back_as_vec[i] << ",";
  }
  cout << endl;

  Dict d;
  d["hi"] = Py::String("3");
  d["hello"] = Py::Int(4);
  d["apple"] = Py::Float(4.4);
  d["banana"] = Py::String("8.8");

  map<string, int> mymap;
  ConvertXY::convert(d.ptr(), mymap);

  for (map<string, int>::iterator it(mymap.begin()); it != mymap.end(); it++) {
    cout << "key: " << it->first << " value: " << it->second << endl;
  }

  PyObject *dict_back = ConvertXY::convert(mymap);
  cout << Py::Object(dict_back) << endl;
  

  //long y;
  //ConvertXY::convert(el.ptr(), y);
  return 1;
}
