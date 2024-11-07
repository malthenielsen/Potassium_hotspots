/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__AMPA
#define _nrn_initial _nrn_initial__AMPA
#define nrn_cur _nrn_cur__AMPA
#define _nrn_current _nrn_current__AMPA
#define nrn_jacob _nrn_jacob__AMPA
#define nrn_state _nrn_state__AMPA
#define _net_receive _net_receive__AMPA 
#define betadyn betadyn__AMPA 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define gbar _p[0]
#define gbar_columnindex 0
#define tau_r _p[1]
#define tau_r_columnindex 1
#define tau_d _p[2]
#define tau_d_columnindex 2
#define qfact _p[3]
#define qfact_columnindex 3
#define ca_ratio _p[4]
#define ca_ratio_columnindex 4
#define ko_change _p[5]
#define ko_change_columnindex 5
#define na_change _p[6]
#define na_change_columnindex 6
#define itmp _p[7]
#define itmp_columnindex 7
#define i _p[8]
#define i_columnindex 8
#define ical _p[9]
#define ical_columnindex 9
#define t1 _p[10]
#define t1_columnindex 10
#define countflag _p[11]
#define countflag_columnindex 11
#define spkcnt _p[12]
#define spkcnt_columnindex 12
#define scale _p[13]
#define scale_columnindex 13
#define eErev _p[14]
#define eErev_columnindex 14
#define y1 _p[15]
#define y1_columnindex 15
#define y2 _p[16]
#define y2_columnindex 16
#define g _p[17]
#define g_columnindex 17
#define ina _p[18]
#define ina_columnindex 18
#define ik _p[19]
#define ik_columnindex 19
#define y1_add _p[20]
#define y1_add_columnindex 20
#define y1_loc _p[21]
#define y1_loc_columnindex 21
#define ki _p[22]
#define ki_columnindex 22
#define ko _p[23]
#define ko_columnindex 23
#define nai _p[24]
#define nai_columnindex 24
#define nao _p[25]
#define nao_columnindex 25
#define Dy1 _p[26]
#define Dy1_columnindex 26
#define Dy2 _p[27]
#define Dy2_columnindex 27
#define v _p[28]
#define v_columnindex 28
#define _g _p[29]
#define _g_columnindex 29
#define _tsav _p[30]
#define _tsav_columnindex 30
#define _nd_area  *_ppvar[0]._pval
#define _ion_ical	*_ppvar[2]._pval
#define _ion_dicaldv	*_ppvar[3]._pval
#define _ion_ko	*_ppvar[4]._pval
#define _ion_ki	*_ppvar[5]._pval
#define _ion_ik	*_ppvar[6]._pval
#define _ion_dikdv	*_ppvar[7]._pval
#define _ion_nao	*_ppvar[8]._pval
#define _ion_nai	*_ppvar[9]._pval
#define _ion_ina	*_ppvar[10]._pval
#define _ion_dinadv	*_ppvar[11]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(Object* _ho) { void* create_point_process(int, Object*);
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt(void*);
 static double _hoc_loc_pnt(void* _vptr) {double loc_point_process(int, void*);
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(void* _vptr) {double has_loc_point(void*);
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(void* _vptr) {
 double get_loc_point_process(void*); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 0, 0
};
 /* declare global and static user variables */
#define Erev Erev_AMPA
 double Erev = 0;
#define saturate saturate_AMPA
 double saturate = 1.2;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "Erev_AMPA", "mV",
 "gbar", "umho",
 "tau_r", "ms",
 "tau_d", "ms",
 "y1", "/ms",
 "itmp", "nA",
 "i", "nA",
 "ical", "nA",
 "t1", "ms",
 "eErev", "mV",
 0,0
};
 static double delta_t = 0.01;
 static double y20 = 0;
 static double y10 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "Erev_AMPA", &Erev_AMPA,
 "saturate_AMPA", &saturate_AMPA,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[12]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"AMPA",
 "gbar",
 "tau_r",
 "tau_d",
 "qfact",
 "ca_ratio",
 "ko_change",
 "na_change",
 0,
 "itmp",
 "i",
 "ical",
 "t1",
 "countflag",
 "spkcnt",
 "scale",
 "eErev",
 0,
 "y1",
 "y2",
 0,
 0};
 static Symbol* _cal_sym;
 static Symbol* _k_sym;
 static Symbol* _na_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 31, _prop);
 	/*initialize range parameters*/
 	gbar = 0.00085;
 	tau_r = 2.2;
 	tau_d = 11.5;
 	qfact = 2;
 	ca_ratio = 0.005;
 	ko_change = 1;
 	na_change = 1;
  }
 	_prop->param = _p;
 	_prop->param_size = 31;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 13, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_cal_sym);
 	_ppvar[2]._pval = &prop_ion->param[3]; /* ical */
 	_ppvar[3]._pval = &prop_ion->param[4]; /* _ion_dicaldv */
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[4]._pval = &prop_ion->param[2]; /* ko */
 	_ppvar[5]._pval = &prop_ion->param[1]; /* ki */
 	_ppvar[6]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[7]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 prop_ion = need_memb(_na_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[8]._pval = &prop_ion->param[2]; /* nao */
 	_ppvar[9]._pval = &prop_ion->param[1]; /* nai */
 	_ppvar[10]._pval = &prop_ion->param[3]; /* ina */
 	_ppvar[11]._pval = &prop_ion->param[4]; /* _ion_dinadv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _net_receive(Point_process*, double*, double);
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _ampa_2_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("cal", 2.0);
 	ion_reg("k", 1.0);
 	ion_reg("na", 1.0);
 	_cal_sym = hoc_lookup("cal_ion");
 	_k_sym = hoc_lookup("k_ion");
 	_na_sym = hoc_lookup("na_ion");
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 31, 13);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "cal_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cal_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 5, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 6, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 7, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 8, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 9, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 10, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 11, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 12, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 2;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 AMPA /home/nordentoft/Documents/Potassium_and_dendrites/fractal_neuron/test_bench/mod_shai/ampa_2.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double FARADAY = 96489.0;
 static double R = 8.314;
static int _reset;
static char *modelname = "AMPA synapse for nucleus accumbens model";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[2], _dlist1[2];
 static int betadyn(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   Dy1 = - y1 / ( tau_d / qfact ) ;
   Dy2 = y1 - y2 / ( tau_r / qfact ) ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 Dy1 = Dy1  / (1. - dt*( ( - 1.0 ) / ( tau_d / qfact ) )) ;
 Dy2 = Dy2  / (1. - dt*( ( - ( 1.0 ) / ( tau_r / qfact ) ) )) ;
  return 0;
}
 /*END CVODE*/
 static int betadyn (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
    y1 = y1 + (1. - exp(dt*(( - 1.0 ) / ( tau_d / qfact ))))*(- ( 0.0 ) / ( ( - 1.0 ) / ( tau_d / qfact ) ) - y1) ;
    y2 = y2 + (1. - exp(dt*(( - ( 1.0 ) / ( tau_r / qfact ) ))))*(- ( y1 ) / ( ( - ( 1.0 ) / ( tau_r / qfact ) ) ) - y2) ;
   }
  return 0;
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t; {
   _args[1] = _args[1] * exp ( - ( t - t1 ) / ( tau_d / qfact ) ) ;
   y1_add = ( 1.0 - _args[1] / saturate ) ;
   _args[1] = _args[1] + y1_add ;
     if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = y1;
    double __primary = (y1 + y1_add) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / ( tau_d / qfact ) ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / ( tau_d / qfact ) ) - __primary );
    y1 += __primary;
  } else {
 y1 = y1 + y1_add ;
     }
 t1 = t ;
   spkcnt = spkcnt + 1.0 ;
   scale = _args[0] ;
   } }
 
static int _ode_count(int _type){ return 2;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ko = _ion_ko;
  ki = _ion_ki;
  nao = _ion_nao;
  nai = _ion_nai;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
    }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 2; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ko = _ion_ko;
  ki = _ion_ki;
  nao = _ion_nao;
  nai = _ion_nai;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_cal_sym, _ppvar, 2, 3);
   nrn_update_ion_pointer(_cal_sym, _ppvar, 3, 4);
   nrn_update_ion_pointer(_k_sym, _ppvar, 4, 2);
   nrn_update_ion_pointer(_k_sym, _ppvar, 5, 1);
   nrn_update_ion_pointer(_k_sym, _ppvar, 6, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 7, 4);
   nrn_update_ion_pointer(_na_sym, _ppvar, 8, 2);
   nrn_update_ion_pointer(_na_sym, _ppvar, 9, 1);
   nrn_update_ion_pointer(_na_sym, _ppvar, 10, 3);
   nrn_update_ion_pointer(_na_sym, _ppvar, 11, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  y2 = y20;
  y1 = y10;
 {
   y1_add = 0.0 ;
   scale = 0.0 ;
   spkcnt = 0.0 ;
   countflag = 0.0 ;
   t1 = 0.0 ;
   y1_loc = 0.0 ;
   }
 
}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _tsav = -1e20;
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
  ko = _ion_ko;
  ki = _ion_ki;
  nao = _ion_nao;
  nai = _ion_nai;
 initmodel(_p, _ppvar, _thread, _nt);
   }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   eErev = - 1000.0 * R * ( celsius + 273.16 ) / ( FARADAY ) * log ( ( ki + nai * na_change ) / ( ( ko * ko_change ) + ( nao ) ) ) ;
   g = gbar * y2 ;
   itmp = scale * g * ( v - eErev ) ;
   i = ( 1.0 - ca_ratio ) * itmp ;
   ical = ca_ratio * itmp ;
   ina = 5.0 * itmp / 10.0 ;
   ik = 3.0 * itmp / 5.0 ;
   }
 _current += i;
 _current += ical;
 _current += ik;
 _current += ina;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
  ko = _ion_ko;
  ki = _ion_ki;
  nao = _ion_nao;
  nai = _ion_nai;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dina;
 double _dik;
 double _dical;
  _dical = ical;
  _dik = ik;
  _dina = ina;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dicaldv += (_dical - ical)/.001 * 1.e2/ (_nd_area);
  _ion_dikdv += (_dik - ik)/.001 * 1.e2/ (_nd_area);
  _ion_dinadv += (_dina - ina)/.001 * 1.e2/ (_nd_area);
 	}
 _g = (_g - _rhs)/.001;
  _ion_ical += ical * 1.e2/ (_nd_area);
  _ion_ik += ik * 1.e2/ (_nd_area);
  _ion_ina += ina * 1.e2/ (_nd_area);
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
  ko = _ion_ko;
  ki = _ion_ki;
  nao = _ion_nao;
  nai = _ion_nai;
 {   betadyn(_p, _ppvar, _thread, _nt);
  }   }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = y1_columnindex;  _dlist1[0] = Dy1_columnindex;
 _slist1[1] = y2_columnindex;  _dlist1[1] = Dy2_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/nordentoft/Documents/Potassium_and_dendrites/fractal_neuron/test_bench/mod_shai/ampa_2.mod";
static const char* nmodl_file_text = 
  "\n"
  "TITLE    AMPA synapse for nucleus accumbens model\n"
  "\n"
  ": see comments below\n"
  "\n"
  "\n"
  "\n"
  "NEURON {\n"
  "\n"
  "	POINT_PROCESS AMPA\n"
  "\n"
  "	RANGE gbar, tau_r, tau_d, scale, spkcnt, countflag, i, t1, ca_ratio, ical, itmp, qfact, ko_change, na_change, eErev\n"
  "\n"
  "	NONSPECIFIC_CURRENT i\n"
  "\n"
  " 	USEION cal WRITE ical VALENCE 2\n"
  "\n"
  "    USEION k READ ko, ki WRITE ik VALENCE 1\n"
  "\n"
  "    USEION na READ nao, nai WRITE ina VALENCE 1\n"
  "\n"
  "\n"
  "\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "UNITS {\n"
  "\n"
  "	(nA) = (nanoamp)\n"
  "\n"
  "	(mV) = (millivolt)\n"
  "\n"
  "	(umho) = (micromho)\n"
  "\n"
  "    FARADAY = 96489 (coul)\n"
  "	R       = 8.314 (volt-coul/degC)\n"
  "\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "PARAMETER {\n"
  "\n"
  "	gbar = 8.5e-4   (umho) 	: approx 0.5:1 NMDA:AMPA ratio (Myme 2003)\n"
  "\n"
  "							:   with mg = 0, vh = -70, one pulse, NMDA = 300 pS\n"
  "\n"
  "							:   here AMPA = 593 pS (NMDA set to Dalby 2003)\n"
  "\n"
  "	tau_r = 2.2 	(ms)   	: Gotz 1997, Table 1 - rise tau\n"
  "\n"
  "	tau_d = 11.5  	(ms)   	: Gotz 1997, Table 1 - decay tau\n"
  "\n"
  "	\n"
  "\n"
  "	Erev = 0    	(mV)   	: reversal potential, Jahn 1998\n"
  "\n"
  "	saturate = 1.2 			: causes the conductance to saturate - matched to \n"
  "\n"
  "							:    Destexhe's reduced model in [1]\n"
  "\n"
  "	qfact = 2				: convert 22 degC to 35 degC\n"
  "\n"
  "	ca_ratio = 0.005			: ratio of calcium current to total current\n"
  "\n"
  "							: Burnashev/Sakmann J Phys 1995 485:403-418\n"
  "\n"
  "							: with Carter/Sabatini Neuron 2004 44:483-493\n"
  "\n"
  "    ko_change = 1\n"
  "    na_change = 1\n"
  "  \n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "ASSIGNED {\n"
  "\n"
  "	g (umho)\n"
  "\n"
  "	v (mV)   		: postsynaptic voltage\n"
  "\n"
  "	itmp	(nA)	: temp value of current\n"
  "\n"
  "	i (nA)   		: nonspecific current = g*(v - Erev)\n"
  "\n"
  "	ical (nA)		: calcium current through AMPA synapse (Carter/Sabatini)\n"
  "\n"
  "	ina (nA)		: sodium current through AMPA synapse \n"
  "	ik (nA)		    : potassium current through AMPA synapse \n"
  "\n"
  "	t1 (ms)\n"
  "\n"
  "	\n"
  "\n"
  "	y1_add (/ms)    : value added to y1 when a presynaptic spike is registered\n"
  "\n"
  "	y1_loc (/ms)\n"
  "\n"
  "\n"
  "\n"
  "	countflag		: start/stop counting spikes delivered\n"
  "\n"
  "	spkcnt			: counts number of events delivered to synapse\n"
  "\n"
  "	scale			: scale allows the current to be scaled by weight\n"
  "\n"
  "					: so NetCon(...,2) gives 2*the current as NetCon(...,1)\n"
  "\n"
  "    ki  (mM)\n"
  "    ko  (mM)\n"
  "    nai  (mM)\n"
  "    nao  (mM)\n"
  "    eErev (mV)\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "STATE { \n"
  "\n"
  "	y1 (/ms) \n"
  "\n"
  "	y2    			: sum of beta-functions, describing the total conductance\n"
  "\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "INITIAL {\n"
  "\n"
  "  	y1_add = 0\n"
  "\n"
  "	scale = 0\n"
  "\n"
  "	spkcnt = 0\n"
  "\n"
  "	countflag = 0\n"
  "\n"
  "	t1 = 0\n"
  "\n"
  "	y1_loc = 0\n"
  "\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "BREAKPOINT {\n"
  "\n"
  "  	SOLVE betadyn METHOD cnexp\n"
  "\n"
  "    eErev = -1000*R*(celsius+273.16) / (FARADAY) * log((ki + nai * na_change)/((ko*ko_change) + (nao))) \n"
  "	g = gbar * y2\n"
  "\n"
  "  	itmp = scale * g * (v - eErev)\n"
  "\n"
  "  	i = (1-ca_ratio) * itmp\n"
  "\n"
  "  	ical = ca_ratio * itmp\n"
  "\n"
  "    ina = 5 * itmp/10\n"
  "\n"
  "    ik =  3 * itmp/5\n"
  "  \n"
  "    :printf(\"Ik %g, Ina %g, erev %g, v %g \\n\", ik, ina, eErev, v)\n"
  "    :printf(\"ki %g, ko %g, nai %g, na0 %g \\n\", ki, ko, nai, nao)\n"
  "\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "DERIVATIVE betadyn {\n"
  "\n"
  "	: dynamics of the beta-function, from [2]\n"
  "\n"
  "	y1' = -y1 / (tau_d/qfact)\n"
  "\n"
  "	y2' = y1 - y2 / (tau_r/qfact)\n"
  "\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "NET_RECEIVE( weight, y1_loc (/ms) ) {\n"
  "\n"
  "	: updating the local y1 variable\n"
  "\n"
  "	y1_loc = y1_loc*exp( -(t - t1) / (tau_d/qfact) )\n"
  "\n"
  "\n"
  "\n"
  "	: y1_add is dependent on the present value of the local\n"
  "\n"
  "	: y1 variable, y1_loc\n"
  "\n"
  "	y1_add = (1 - y1_loc/saturate)\n"
  "\n"
  "\n"
  "\n"
  "	: update the local y1 variable\n"
  "\n"
  "	y1_loc = y1_loc + y1_add\n"
  "\n"
  "\n"
  "\n"
  "	: presynaptic spike is finaly registered\n"
  "\n"
  "	y1 = y1 + y1_add\n"
  "\n"
  "\n"
  "\n"
  "	: store the spike time\n"
  "\n"
  "	t1 = t\n"
  "\n"
  "\n"
  "\n"
  "	spkcnt = spkcnt + 1\n"
  "\n"
  "\n"
  "\n"
  "	scale = weight\n"
  "\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  ;
#endif
