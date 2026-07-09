/*
 * asv_bench_mpi._native
 *
 * Free-threaded CPython extension: heap types (PyType_FromSpec) + GIL-released
 * system launches (fork/execve/waitpid) and native kernel calls (dlopen/dlsym).
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

static double
mono_seconds(void)
{
    struct timespec ts;
#if defined(CLOCK_MONOTONIC)
    clock_gettime(CLOCK_MONOTONIC, &ts);
#else
    clock_gettime(CLOCK_REALTIME, &ts);
#endif
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

typedef double (*asv_kernel_fn)(long);

/* ================================================================== */
/* NativeKernel heap type                                             */
/* ================================================================== */

typedef struct {
    PyObject_HEAD
    void *handle;
    asv_kernel_fn fn;
    char *so_path;
    char *symbol;
} NativeKernelObject;

typedef struct {
    PyObject *NativeKernel_Type;
} native_state;

#define NATIVE_STATE(m) ((native_state *)PyModule_GetState(m))

static int
NativeKernel_clear(PyObject *self)
{
    (void)self;
    return 0;
}

static int
NativeKernel_traverse(PyObject *self, visitproc visit, void *arg)
{
    (void)self;
    (void)visit;
    (void)arg;
    return 0;
}

static void
NativeKernel_finalize(PyObject *self)
{
    NativeKernelObject *op = (NativeKernelObject *)self;
    if (op->handle) {
        dlclose(op->handle);
        op->handle = NULL;
        op->fn = NULL;
    }
    free(op->so_path);
    op->so_path = NULL;
    free(op->symbol);
    op->symbol = NULL;
}

static void
NativeKernel_dealloc(PyObject *self)
{
    PyObject_GC_UnTrack(self);
    NativeKernel_finalize(self);
    {
        PyTypeObject *tp = Py_TYPE(self);
        freefunc free_fn = (freefunc)PyType_GetSlot(tp, Py_tp_free);
        free_fn(self);
        Py_DECREF(tp);
    }
}

static PyObject *
NativeKernel_new(PyTypeObject *type, PyObject *args, PyObject *kw)
{
    allocfunc alloc = (allocfunc)PyType_GetSlot(type, Py_tp_alloc);
    NativeKernelObject *self;
    (void)args;
    (void)kw;
    self = (NativeKernelObject *)alloc(type, 0);
    if (self == NULL)
        return NULL;
    self->handle = NULL;
    self->fn = NULL;
    self->so_path = NULL;
    self->symbol = NULL;
    return (PyObject *)self;
}

static int
NativeKernel_init(PyObject *self, PyObject *args, PyObject *kw)
{
    NativeKernelObject *op = (NativeKernelObject *)self;
    const char *so_path = NULL;
    const char *symbol = NULL;
    static char *kwlist[] = {"so_path", "symbol", NULL};
    void *handle;
    void *sym;
    char *err;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "ss:NativeKernel", kwlist,
                                     &so_path, &symbol))
        return -1;

    NativeKernel_finalize(self);

    handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    if (handle == NULL) {
        PyErr_Format(PyExc_OSError, "dlopen(%s): %s", so_path, dlerror());
        return -1;
    }
    dlerror();
    sym = dlsym(handle, symbol);
    err = dlerror();
    if (err != NULL || sym == NULL) {
        dlclose(handle);
        PyErr_Format(PyExc_OSError, "dlsym(%s, %s): %s", so_path, symbol,
                     err ? err : "NULL");
        return -1;
    }

    op->handle = handle;
    op->fn = (asv_kernel_fn)sym;
    op->so_path = strdup(so_path);
    op->symbol = strdup(symbol);
    if (op->so_path == NULL || op->symbol == NULL) {
        NativeKernel_finalize(self);
        PyErr_NoMemory();
        return -1;
    }
    return 0;
}

static PyObject *
NativeKernel_call(PyObject *self, PyObject *args, PyObject *kw)
{
    NativeKernelObject *op = (NativeKernelObject *)self;
    long n = 0;
    double result;
    static char *kwlist[] = {"n", NULL};
    asv_kernel_fn fn;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "|l:call", kwlist, &n))
        return NULL;
    if (op->fn == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "NativeKernel not initialized");
        return NULL;
    }
    fn = op->fn;
    Py_BEGIN_ALLOW_THREADS
    result = fn(n);
    Py_END_ALLOW_THREADS
    return PyFloat_FromDouble(result);
}

static PyObject *
NativeKernel_time(PyObject *self, PyObject *args, PyObject *kw)
{
    NativeKernelObject *op = (NativeKernelObject *)self;
    long n = 0;
    double t0, t1, result;
    static char *kwlist[] = {"n", NULL};
    asv_kernel_fn fn;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "|l:time", kwlist, &n))
        return NULL;
    if (op->fn == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "NativeKernel not initialized");
        return NULL;
    }
    fn = op->fn;
    Py_BEGIN_ALLOW_THREADS
    t0 = mono_seconds();
    result = fn(n);
    t1 = mono_seconds();
    (void)result;
    Py_END_ALLOW_THREADS
    return PyFloat_FromDouble(t1 - t0);
}

static PyObject *
NativeKernel_repr(PyObject *self)
{
    NativeKernelObject *op = (NativeKernelObject *)self;
    return PyUnicode_FromFormat("NativeKernel(%s, %s)",
                                op->so_path ? op->so_path : "?",
                                op->symbol ? op->symbol : "?");
}

static PyMethodDef NativeKernel_methods[] = {
    {"call", (PyCFunction)NativeKernel_call, METH_VARARGS | METH_KEYWORDS,
     "call(n=0) -> float — invoke double(*)(long) with GIL released"},
    {"time", (PyCFunction)NativeKernel_time, METH_VARARGS | METH_KEYWORDS,
     "time(n=0) -> float — wall seconds, GIL released"},
    {NULL, NULL, 0, NULL}
};

static PyType_Slot NativeKernel_Type_slots[] = {
    {Py_tp_dealloc, (void *)NativeKernel_dealloc},
    {Py_tp_traverse, (void *)NativeKernel_traverse},
    {Py_tp_clear, (void *)NativeKernel_clear},
    {Py_tp_finalize, (void *)NativeKernel_finalize},
    {Py_tp_methods, NativeKernel_methods},
    {Py_tp_init, (void *)NativeKernel_init},
    {Py_tp_new, (void *)NativeKernel_new},
    {Py_tp_repr, (void *)NativeKernel_repr},
    {0, 0},
};

static PyType_Spec NativeKernel_Type_spec = {
    "asv_bench_mpi._native.NativeKernel",
    sizeof(NativeKernelObject),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HEAPTYPE,
    NativeKernel_Type_slots
};

/* ================================================================== */
/* fork/exec helpers                                                  */
/* ================================================================== */

static void
free_cstr_array(char **arr)
{
    Py_ssize_t i;
    if (arr == NULL)
        return;
    for (i = 0; arr[i] != NULL; ++i)
        free(arr[i]);
    free(arr);
}

static char **
dup_argv_from_seq(PyObject *seq)
{
    Py_ssize_t n, i;
    char **argv;

    n = PySequence_Size(seq);
    if (n < 1) {
        PyErr_SetString(PyExc_ValueError, "argv must be non-empty");
        return NULL;
    }
    argv = (char **)calloc((size_t)n + 1, sizeof(char *));
    if (argv == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    for (i = 0; i < n; ++i) {
        PyObject *item = PySequence_GetItem(seq, i);
        const char *s;
        if (item == NULL) {
            free_cstr_array(argv);
            return NULL;
        }
        s = PyUnicode_AsUTF8(item);
        if (s == NULL) {
            Py_DECREF(item);
            free_cstr_array(argv);
            return NULL;
        }
        argv[i] = strdup(s);
        Py_DECREF(item);
        if (argv[i] == NULL) {
            free_cstr_array(argv);
            PyErr_NoMemory();
            return NULL;
        }
    }
    argv[n] = NULL;
    return argv;
}

static char **
dup_envp_from_dict(PyObject *env_obj)
{
    PyObject *merged;
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    Py_ssize_t n, i;
    char **envp;
    char **ep;

    if (env_obj == NULL || env_obj == Py_None)
        return NULL;
    if (!PyDict_Check(env_obj)) {
        PyErr_SetString(PyExc_TypeError, "env must be a dict or None");
        return NULL;
    }

    merged = PyDict_New();
    if (merged == NULL)
        return NULL;
    for (ep = environ; ep && *ep; ++ep) {
        char *eq = strchr(*ep, '=');
        PyObject *k, *v;
        if (!eq)
            continue;
        k = PyUnicode_FromStringAndSize(*ep, (Py_ssize_t)(eq - *ep));
        v = PyUnicode_FromString(eq + 1);
        if (k == NULL || v == NULL) {
            Py_XDECREF(k);
            Py_XDECREF(v);
            Py_DECREF(merged);
            return NULL;
        }
        if (PyDict_SetItem(merged, k, v) < 0) {
            Py_DECREF(k);
            Py_DECREF(v);
            Py_DECREF(merged);
            return NULL;
        }
        Py_DECREF(k);
        Py_DECREF(v);
    }
    if (PyDict_Update(merged, env_obj) < 0) {
        Py_DECREF(merged);
        return NULL;
    }

    n = PyDict_Size(merged);
    envp = (char **)calloc((size_t)n + 1, sizeof(char *));
    if (envp == NULL) {
        Py_DECREF(merged);
        PyErr_NoMemory();
        return NULL;
    }
    i = 0;
    pos = 0;
    while (PyDict_Next(merged, &pos, &key, &value)) {
        const char *ks = PyUnicode_AsUTF8(key);
        const char *vs = PyUnicode_AsUTF8(value);
        size_t len;
        char *line;
        if (ks == NULL || vs == NULL) {
            free_cstr_array(envp);
            Py_DECREF(merged);
            return NULL;
        }
        len = strlen(ks) + 1 + strlen(vs) + 1;
        line = (char *)malloc(len);
        if (line == NULL) {
            free_cstr_array(envp);
            Py_DECREF(merged);
            PyErr_NoMemory();
            return NULL;
        }
        snprintf(line, len, "%s=%s", ks, vs);
        envp[i++] = line;
    }
    envp[i] = NULL;
    Py_DECREF(merged);
    return envp;
}

/*
 * run_executable(argv, env=None, cwd=None) -> dict
 *
 * SYSTEM CALLS (GIL released for the whole fork/wait block):
 *   fork, execve/execvp, waitpid, chdir, clock_gettime
 */
static PyObject *
native_run_executable(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject *argv_obj = NULL;
    PyObject *env_obj = Py_None;
    PyObject *cwd_obj = Py_None;
    static char *kwlist[] = {"argv", "env", "cwd", NULL};
    char **argv = NULL;
    char **envp = NULL;
    char *cwd = NULL;
    double t0, t1;
    pid_t pid;
    int status = 0;
    int fork_errno = 0;
    int wait_errno = 0;
    int rc_kind = 0;
    int exit_code = 0;
    PyObject *result;

    (void)self;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|OO:run_executable", kwlist,
                                     &argv_obj, &env_obj, &cwd_obj))
        return NULL;
    if (!PyList_Check(argv_obj) && !PyTuple_Check(argv_obj)) {
        PyErr_SetString(PyExc_TypeError, "argv must be a list or tuple of str");
        return NULL;
    }

    argv = dup_argv_from_seq(argv_obj);
    if (argv == NULL)
        return NULL;

    if (env_obj != Py_None) {
        envp = dup_envp_from_dict(env_obj);
        if (envp == NULL && PyErr_Occurred())
            goto fail;
    }

    if (cwd_obj != Py_None) {
        const char *cs;
        if (!PyUnicode_Check(cwd_obj)) {
            PyErr_SetString(PyExc_TypeError, "cwd must be str or None");
            goto fail;
        }
        cs = PyUnicode_AsUTF8(cwd_obj);
        if (cs == NULL)
            goto fail;
        cwd = strdup(cs);
        if (cwd == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
    }

    t0 = mono_seconds();
    Py_BEGIN_ALLOW_THREADS
    pid = fork();
    if (pid == 0) {
        if (cwd != NULL && chdir(cwd) != 0)
            _exit(126);
        if (envp != NULL)
            execve(argv[0], argv, envp);
        else
            execvp(argv[0], argv);
        _exit(127);
    } else if (pid < 0) {
        fork_errno = errno;
        rc_kind = 3;
    } else {
        if (waitpid(pid, &status, 0) < 0) {
            wait_errno = errno;
            rc_kind = 2;
        } else if (WIFEXITED(status)) {
            exit_code = WEXITSTATUS(status);
            if (exit_code != 0)
                rc_kind = 1;
        } else {
            rc_kind = 1;
            exit_code = -1;
        }
    }
    t1 = mono_seconds();
    Py_END_ALLOW_THREADS

    if (rc_kind == 3) {
        errno = fork_errno;
        PyErr_SetFromErrno(PyExc_OSError);
        goto fail;
    }
    if (rc_kind == 2) {
        errno = wait_errno;
        PyErr_SetFromErrno(PyExc_OSError);
        goto fail;
    }

    result = Py_BuildValue("{s:d,s:i,s:i,s:s}",
                           "wall_seconds", t1 - t0,
                           "returncode", exit_code,
                           "ok", rc_kind == 0 ? 1 : 0,
                           "argv0", argv[0]);
    free_cstr_array(argv);
    free_cstr_array(envp);
    free(cwd);
    return result;

fail:
    free_cstr_array(argv);
    free_cstr_array(envp);
    free(cwd);
    return NULL;
}

static PyObject *
native_extension_flags(PyObject *self, PyObject *Py_UNUSED(args))
{
    PyObject *d = PyDict_New();
    PyObject *api = NULL;
    PyObject *sysc = NULL;
    int gil_not_used = 0;
    (void)self;
#ifdef Py_mod_gil
    gil_not_used = 1;
#endif
    if (d == NULL)
        return NULL;
    if (PyDict_SetItemString(d, "gil_not_used", gil_not_used ? Py_True : Py_False) < 0)
        goto err;
    if (PyDict_SetItemString(d, "heap_type", Py_True) < 0)
        goto err;
    if (PyDict_SetItemString(d, "py_begin_allow_threads", Py_True) < 0)
        goto err;
    if (PyDict_SetItemString(d, "run_executable_applies_env", Py_True) < 0)
        goto err;
    api = PyUnicode_FromString("native-launch+exec");
    sysc = PyUnicode_FromString(
        "fork,execve/execvp,waitpid,chdir,dlopen,dlsym,clock_gettime");
    if (api == NULL || sysc == NULL)
        goto err;
    if (PyDict_SetItemString(d, "api", api) < 0)
        goto err;
    if (PyDict_SetItemString(d, "system_calls", sysc) < 0)
        goto err;
    Py_DECREF(api);
    Py_DECREF(sysc);
    return d;
err:
    Py_XDECREF(api);
    Py_XDECREF(sysc);
    Py_DECREF(d);
    return NULL;
}

static PyMethodDef module_methods[] = {
    {"run_executable", (PyCFunction)native_run_executable,
     METH_VARARGS | METH_KEYWORDS,
     "run_executable(argv, env=None, cwd=None) -> dict\n"
     "fork+execve/execvp+waitpid in C; GIL released; env merges os.environ."},
    {"extension_flags", (PyCFunction)native_extension_flags, METH_NOARGS,
     "extension_flags() -> dict"},
    {NULL, NULL, 0, NULL}
};

static int
native_modexec(PyObject *m)
{
    native_state *state = NATIVE_STATE(m);
    state->NativeKernel_Type = PyType_FromSpec(&NativeKernel_Type_spec);
    if (state->NativeKernel_Type == NULL)
        return -1;
    Py_INCREF(state->NativeKernel_Type);
    if (PyModule_AddObject(m, "NativeKernel", state->NativeKernel_Type) < 0) {
        Py_DECREF(state->NativeKernel_Type);
        Py_CLEAR(state->NativeKernel_Type);
        return -1;
    }
    return 0;
}

static int
native_traverse(PyObject *module, visitproc visit, void *arg)
{
    native_state *state = NATIVE_STATE(module);
    Py_VISIT(state->NativeKernel_Type);
    return 0;
}

static int
native_clear(PyObject *module)
{
    native_state *state = NATIVE_STATE(module);
    Py_CLEAR(state->NativeKernel_Type);
    return 0;
}

static void
native_free(void *module)
{
    (void)native_clear((PyObject *)module);
}

static PyModuleDef_Slot native_slots[] = {
    {Py_mod_exec, (void *)native_modexec},
#ifdef Py_mod_gil
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL}
};

PyDoc_STRVAR(module_doc,
"Native launch: dlopen kernels + fork/exec system launches (GIL released).");

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "asv_bench_mpi._native",
    module_doc,
    sizeof(native_state),
    module_methods,
    native_slots,
    native_traverse,
    native_clear,
    native_free
};

PyMODINIT_FUNC
PyInit__native(void)
{
    return PyModuleDef_Init(&moduledef);
}
