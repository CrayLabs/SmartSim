from fortran_code_generator import fortran_argument
from ctypes import c_float, c_double

# Unit tests
#scalar

indent = '  '
class fortran_program:
    def __init__(self, name):
        self.name = name
        self.tests = []
    def write_source(self,exact_key=False,file=None):
        out  = f'program {self.name}\n'
        out += indent + f'use iso_c_binding, only : c_ptr\n'
        out += indent + f'use mpi\n'
        out += indent + f'use client_fortran_api\n'
        out += indent + f'use unit_test_aux\n\n'
        out += indent + f'implicit none\n\n'
        for test in self.tests:
            out += test.write_declaration()

        out += indent + 'integer :: pe_id\n'
        out += indent + 'integer :: err_code\n'
        out += indent + 'character(len=9) :: key_prefix\n'
        out += indent + 'type(c_ptr) :: smartsim_client\n\n'
        out += indent + 'call MPI_init( err_code )\n'
        out += indent + (
            'call MPI_comm_rank( MPI_COMM_WORLD, pe_id, err_code )\n')
        out += indent + 'write(key_prefix, "(A,I6.6)") "pe_",pe_id\n'
        out += indent + 'num_failed = 0\n'
        out += indent + 'smartsim_client = init_ssc_client()\n\n'

        for test in self.tests:
            out += test.write_test(exact_key)
            out += '\n'

        out += indent + 'call MPI_Finalize(err_code)\n'
        out += indent + 'if (num_failed > 0) stop -1\n'
        out += f'end program {self.name}'
        print(out,file=file)

class scalar_unit_test:
    def __init__(self,vartype,kind):
        self.vartype = vartype
        self.kind = kind
        self.key_name = f'key_prefix//"test_scalar_{vartype}_{kind}"'
        self.true_var = fortran_argument(
                vartype, f'true_scalar_{vartype}_{kind}', kind=kind)
        self.recv_var = fortran_argument(
                vartype, f'recv_scalar_{vartype}_{kind}', kind=kind)

        self.true_value = 3
        if (self.vartype) == 'real':
            self.true_value += 0.125

    def write_declaration(self):
        out  = indent+self.true_var.declaration + '\n'
        out += indent+self.recv_var.declaration + '\n'
        return out
    def write_test(self,exact_key=False):
        if exact_key:
            function_modifier = 'exact_key_'
        else:
            function_modifier = ''
        out  = indent + f'{self.true_var.name} = {self.true_value}\n'
        out += indent + (
            f'call put_{function_modifier}scalar(smartsim_client,'+
            f'{self.key_name},{self.true_var.name})\n')
        out += indent + f'call get_{function_modifier}scalar(smartsim_client,{self.key_name},{self.recv_var.name})\n'
        out += indent + f'call check_value({self.true_var.name},{self.recv_var.name},"put/get_"//{self.key_name})\n'
        return out

class array_unit_test:
    def __init__(self,vartype,kind):
        self.vartype = vartype
        self.kind = kind

    def write_declaration(self):
        out  = indent+self.true_var.declaration + '\n'
        out += indent+self.recv_var.declaration + '\n'
        out += indent+self.random_var.declaration + '\n'
        return out

    def write_test(self,exact_key=False):
        if exact_key:
            function_modifier = 'exact_key_'
        else:
            function_modifier = ''

        zero_string = f'{self.vartype}{self.kind}_0'
        out  = indent + f'call RANDOM_NUMBER({self.random_var.name})\n'
        out += indent + f'{self.true_var.name} = {self.random_var.name}*1000\n'
        out += indent + f'call put_{function_modifier}array(smartsim_client,'\
            f'{self.key_name},{self.true_var.name})\n'
        out += indent
        out += f'call get_{function_modifier}array(smartsim_client,'\
            f'{self.key_name},{self.recv_var.name})\n'
        out += indent
        out += f'call check_value({zero_string},SUM({self.true_var.name}-{self.recv_var.name}),"put/get_{function_modifier}"//{self.key_name})\n'
        return out
class array_1d_unit_test(array_unit_test):
    def __init__(self, *args, dim_len = 10):
        super().__init__(*args)
        self.key_name = f'key_prefix//"test_array_1d_{vartype}_{kind}"'
        self.true_var = fortran_argument(
            vartype, f'true_array_1d_{vartype}_{kind}',
            kind=kind, attributes=[f'dimension({dim_len})'])
        self.recv_var = fortran_argument(
            vartype, f'recv_array_1d_{vartype}_{kind}',
            kind=kind, attributes=[f'dimension({dim_len})'])
        self.random_var = fortran_argument(
            'real', f'random_array_1d_for_{vartype}_{kind}',
            kind=kind, attributes=[f'dimension({dim_len})'])

class array_2d_unit_test(array_unit_test):
    def __init__(self, *args, dim1_len = 10, dim2_len = 5):
        super(array_2d_unit_test,self).__init__(*args)
        self.key_name = f'key_prefix//"test_array_2d_{vartype}_{kind}"'
        self.true_var = fortran_argument(
            vartype, f'true_array_2d_{vartype}_{kind}',
            kind=kind, attributes=[f'dimension({dim1_len},{dim2_len})'])
        self.recv_var = fortran_argument(
            vartype, f'recv_array_2d_{vartype}_{kind}',
            kind=kind, attributes=[f'dimension({dim1_len},{dim2_len})'])
        self.random_var = fortran_argument(
            'real', f'random_array_2d_for_{vartype}_{kind}',
            kind=kind, attributes=[f'dimension({dim1_len},{dim2_len})'])

class poll_key_and_check_scalar_unit_test:
    def __init__(self,vartype,kind):
        self.vartype = vartype
        self.kind = kind
        self.true_value = 3
        if vartype == 'real':
            self.true_value += 0.125
        self.false_value = self.true_value + 1
        self.key_name = ('key_prefix//"poll_key_and_check_scalar' +
            f'_{vartype}_{kind}"')
        self.true_var = fortran_argument(
            vartype, f'true_{vartype}_{kind}', kind=kind)
        self.false_var = fortran_argument(
            vartype, f'false_{vartype}_{kind}', kind=kind)
        self.recv_var = fortran_argument(
            'logical',f'result_poll_key_and_check_scalar_{vartype}_{kind}')

    def write_declaration(self):
        out = indent+self.recv_var.declaration + '\n'
        out += indent+self.true_var.declaration + '\n'
        out += indent+self.false_var.declaration + '\n'
        return out
    def write_test(self,exact_key=False):
        if exact_key:
            function_modifier = 'exact_'
            function_modifier2 = 'exact_key_'
        else:
            function_modifier = ''
            function_modifier2 = ''
        out =  indent + f'{self.true_var.name} = {self.true_value}\n'
        out += indent + f'{self.false_var.name} = {self.false_value}\n'
        out += (indent +
            f'call put_{function_modifier2}scalar(smartsim_client,' +
            f'{self.key_name},{self.true_var.name})\n')
        out += (indent +
            f'{self.recv_var.name} = &\n{indent*4}' +
            f'poll_{function_modifier}key_and_check_scalar' +
            f'(smartsim_client,{self.key_name},{self.false_var.name},10,1)\n')
        out += (indent +
            f'call check_value(.false.,{self.recv_var.name},&\n' +
            f'{indent*4}TRIM({self.key_name})//"_expect_false")\n')
        out += (indent +
            f'{self.recv_var.name} = &\n' +
            f'{indent*4}poll_{function_modifier}key_and_check_scalar' +
            f'(smartsim_client,{self.key_name},{self.true_var.name},10,1)\n')
        out += (indent +
            f'call check_value(.true.,&\n{indent*4}{self.recv_var.name},&\n' +
            f'{indent*4}{self.key_name}//"_expect_true")\n')
        return out

vartypes = ['real','integer']
kinds = ['4','8']

tests = [scalar_unit_test,
         array_1d_unit_test,
         array_2d_unit_test,
         poll_key_and_check_scalar_unit_test]

for test in tests:
    fname = f'../smartsim/tests/clients/fortran/{test.__name__}.F90'
    with open(fname,'w') as source_file:
        tests_by_type = []
        unit_test = fortran_program(test.__name__)
        for vartype in vartypes:
            for kind in kinds:
                unit_test.tests.append( test(vartype,kind) )
        unit_test.write_source(file=source_file)
    fname = f'../smartsim/tests/clients/fortran/exact_key_{test.__name__}.F90'
    with open(fname,'w') as source_file:
        tests_by_type = []
        unit_test = fortran_program('exact_key_' + test.__name__)
        for vartype in vartypes:
            for kind in kinds:
                unit_test.tests.append( test(vartype,kind) )
        unit_test.write_source(file=source_file,exact_key=True)
