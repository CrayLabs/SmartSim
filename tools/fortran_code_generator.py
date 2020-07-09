from argparse import ArgumentParser

iso_c_binding_types = ['c_ptr', 'c_loc', 'c_char', 'c_double',
                       'c_int', 'c_float', 'c_null_char', 'c_f_pointer',
                       'c_int32_t', 'c_int64_t']
indent = '  '
class fortran_argument:
    """Describes a Fortran argument
    """
    def __init__(self, ftype, name, description=None, attributes=None, kind=None,
                 length=None, intent = None):
        """Define all descriptors of the Fortran argument

        :param ftype: The type of argument
        :type ftype: str
        :param name: Argument name
        :type name: str
        :param description: A description for the variable
        :type description: str
        :param attributes: Fortran attributes for this variable (e.g. dimension)
        :type attributes: list
        :param kind: The kind of the base Fortran type, defaults to None
        :type kind: str
        :param length: Length of a character string
        :type length: optional
        :param intent: The intent of the variable: in/inout/out, defaults to
                       None which is in
        :type intent: str, optional
        """
        self.ftype = ftype
        self.name  = name
        self.description = description
        self.attributes = attributes
        self.kind = kind
        self.length = length
        self.declaration = None
        self.intent = intent
        if intent:
            if intent == 'in':
                self.intent = 'in   '
            if intent == 'out':
                self.intent = '  out'
        self._declaration_writer()

    def _declaration_writer(self):
        """Writes the Fortran declaration for the argument"""

        declaration_space = 45 # Space at which to start the name declaration
        docstring_space = 80 # Space at which to start the docstring

        self.declaration = self.ftype

        modifier = []
        if self.kind:
            modifier.append(f'kind={self.kind}')
        if self.length and self.kind != 'c_char':
            modifier.append(f'len={self.length}')
        if modifier:
            modifier = ','.join(modifier)
            self.declaration += f'({modifier})'
        if self.attributes:
            self.declaration += ', ' + ', '.join(self.attributes)
        if self.intent:
            self.declaration+= ','
        self.declaration += ' '*max(declaration_space-len(self.declaration), 2 )
        if self.intent:
            self.declaration += f'intent({self.intent})'
        else:
            self.declaration += f'             '
        self.declaration += f' :: {self.name}'
        if self.length and self.kind == 'c_char':
            self.declaration += f'({self.length})'
        if self.description:
            self.declaration += ' '*max(docstring_space-len(self.declaration),2)
            self.declaration += f'!< {self.description}'

class fortran_procedure:
    """Contains groups of Fortran functions/subroutines"""
    def __init__(self, name, description, overload=True):
        """Define all the descriptors and members of the procedure

        :param name: Name to be used for the generic interface
        :type name: str
        :param description: Brief description of the use for the interface
        :type description: str
        :param overload: If true, creates an overloaded procedure that includes
            of its members , defaults to True
        :type overload: bool, optional
        """
        self.name = name
        self.description = description
        self.members = []
        self.overload = overload

    def public_listing(self, file=None):
        """Declares each member and potentially overloaded public

        :param file: Name of the file to right the declaration to,
        defaults to None
        :type file: [type], optional
        """
        for member in self.members:
            print(f'public :: {member.func_basename}',file=file)
        if self.overload:
            print(f'public :: {self.name}', file=file)

    def procedure_declaration(self,file=None):
        """Write the overloaded procedure if previously requested

        :param file: Name of the file to right the declaration to,
        defaults to None
        :type file: [type], optional
        """
        if self.overload:
            contents = f'!> {self.description}\n'
            member_list = f',&\n{indent*2}'.join([member.func_basename
                for member in self.members])
            contents += f'interface {self.name}\n'
            contents += indent + f'module procedure {member_list}\n'
            contents += f'end interface {self.name}\n'
            print(contents,file=file)

    def __iter__(self):
        return iter(self.members)

class fortran_routine:
    """Define the descriptors used to generate Fortran function or subroutine
    """
    def __init__(self, func_type, func_basename, description, f_args, c_args,
                 c_result = None, result = None, c_target = None):
        """[summary]

        :param func_type: The type of the Fortran routine, either 'function' or
                          'subroutine'
        :type func_type: str
        :param func_basename: The name of the routine
        :type func_basename: str
        :param description: A brief description of the routine's purpose
        :type description: str
        :param f_args: List of fortran input arguments
        :type f_args: list
        :param c_args: List of arguments for the C-bound interface
        :type c_args: list
        :param c_result: The return type of the C-bound interface,
                         defaults to None
        :type c_result: fortran_argument, optional
        :param result: Return type of the Fortran function, defaults to None
        :type result: fortran_argument, optional
        :param c_target: The name of the C function that the interface
                        should target, defaults to None meaning that the name
                        will be {basename}_c
        :type c_target: str, optional
        """
        self.func_type = func_type
        self.func_basename = func_basename
        try:
            self.description = description.format(vartype=vartype)
        except:
            self.description = description
        self.f_args = f_args
        self.c_args = c_args
        self.result = result
        self.c_result = c_result
        if c_target:
            self.c_target = c_target
        else:
            self.c_target = f'{func_basename}_c'
        self.interface_name = f'{self.func_basename}_ssc'

    def write_c_interface(self, file = None):
        """Writes the C-bound interface from Fortran to a C-function
        """
        contents = "interface\n"
        arg_list = ', '.join([arg.name for arg in self.c_args])

        if self.c_result:
            contents += indent
            contents += f'{self.c_result.ftype}(kind={self.c_result.kind})' \
                        f'function {self.interface_name}( {arg_list} ) &\n'
        else:
            contents += indent
            contents += f'{self.func_type} {self.interface_name}( {arg_list} )'
            contents += '&\n'

        contents += indent*4 + f'bind(c, name="{self.c_target}")'
        if self.result:
            contents += ' &\n'
            contents += indent*4 + f'result({self.result.name})\n'
        else:
            contents += '\n'

        # Determine which c types we have to import
        c_type_list = []
        for c_type in iso_c_binding_types:
            for arg in self.c_args:
                if c_type in arg.ftype:
                    c_type_list.append(c_type)
                if arg.kind:
                    if c_type == arg.kind:
                        c_type_list.append(c_type)
        if self.c_result:
            c_type_list.append(self.c_result.kind)
        if c_type_list:
            c_types = ', '.join(list(set(c_type_list)))
            contents += indent*2 + f'use iso_c_binding, only : {c_types}\n'
        for arg in self.c_args:
            contents += indent*2 + f'{arg.declaration}\n'
        if self.c_result:
            contents += indent + f'end function {self.func_basename}_ssc\n'
        else:
            contents += indent
            contents += f'end {self.func_type} {self.func_basename}_ssc\n'
        contents += 'end interface'
        print(contents, file=file)

    def write_fortran_routine(self, file = None):
        """Writes the content of the Fortran user-facing routines"""

        contents = indent + f'!> {self.description}\n'
        arg_list = ', '.join([arg.name for arg in self.f_args])
        contents += indent
        contents += f'{self.func_type} {self.func_basename}( {arg_list} )'
        if self.result:
            contents += ' &\n'
            contents += indent*4 + f'result({self.result.name})\n'
        else:
            contents += '\n'
        for arg in self.f_args:
            contents += indent*2 + f'{arg.declaration}\n'
        if self.result:
            contents += indent*2 + f'{self.result.declaration}\n'

        contents += indent*2 + '! Local variables\n'
        contents += indent*2
        contents += 'character(kind=c_char) :: c_key(len(trim(key))+1)\n\n'

        is_array = False
        for arg in self.f_args:
            if arg.attributes:
                is_array = is_array or 'dimension(..)' in arg.attributes
        c_arg_list = []
        for arg in self.c_args:
            if arg.name == 'dims':
                c_arg_list.append('dims(1:ndims)')
            else:
                c_arg_list.append(arg.name)
        c_arg_list = ', '.join(c_arg_list)

        subroutine_calling_functions = ['get_scalar','get_exact_key_scalar']

        ## The following handles special cases for each type of function
        if is_array:
            # Arrays need to have their sizes calculated and a pointer made
            #  to the first element
            contents += indent*2 + 'type(c_ptr) :: array_ptr\n'
            contents += indent*2 + 'integer :: ndims, i\n'
            contents += indent*2
            contents += 'integer(kind=c_int), dimension(MAX_RANK) :: dims\n'

            contents += indent*2
            contents += '! Store the shape of the arrays in reverse order\n'
            contents += indent*2 + 'ndims = size(shape(array))\n'
            contents += indent*2 + 'do i=1,ndims\n'
            contents += indent*2 + '  dims(i) = size(array,ndims+1-i)\n'
            contents += indent*2 + 'enddo\n'
            contents += indent*2 + 'array_ptr = c_loc(array)\n'
            contents += indent*2 + 'c_key = make_c_string(key)\n'

            contents += indent*2 + f'call {self.interface_name}({c_arg_list})\n'
        elif 'function' in self.func_type:
            contents += indent*2 + 'c_key = make_c_string(key)\n'
            contents += indent*2
            contents += f'{self.result.name}' \
                f' = {self.interface_name}({c_arg_list})\n'
        elif any([ name in self.func_basename
                   for name in subroutine_calling_functions ]):
            out_name = [ arg.name for
                         arg in self.f_args if arg.intent.strip() == 'out' ][0]
            contents += indent*2 + 'c_key = make_c_string(key)\n'
            contents += indent*2
            contents += f'{out_name} = {self.interface_name}({c_arg_list})\n'
        else:
            contents += indent*2 + 'c_key = make_c_string(key)\n'
            contents += indent*2 + f'call {self.interface_name}({c_arg_list})\n'
        contents += indent + f'end {self.func_type} {self.func_basename}\n'
        print(contents, file=file)

if __name__ == "__main__":
    # Parse the optional runtime arguments
    parser = ArgumentParser(
        description= "Generate Fortran interfaces and methods for SmartSim")
    parser.add_argument('--outdir', default='../smartsim/clients/')
    args = parser.parse_args()

    # Define all the common arguments used in various routines
    ssc = fortran_argument('type(c_ptr)',
                       'ssc_obj',
                       'Pointer to initialized SmartSim client',
                       attributes = ['value'],
                       intent='in')
    c_key = fortran_argument('character',
                           'c_key',
                           'Key used in the database for the object',
                           kind='c_char',
                           length='*',
                           intent='in')
    f_key = fortran_argument('character',
                           'key',
                           'Key used in the database for the object',
                           length='*',
                           intent='in')
    array_ptr = fortran_argument('type(c_ptr)',
                                 'array_ptr',
                                 'Pointer to the array',
                                 attributes = ['value'],
                                 intent='in')
    dims = fortran_argument('integer',
                            'dims',
                            'Length along each array dimensions',
                            attributes=['dimension(:)'],
                            kind='c_int',
                            intent='in')
    ndims = fortran_argument('integer',
                            'ndims',
                            'Number of dimensions in array',
                            kind='c_int',
                            intent='in')
    poll_frequency = fortran_argument('integer',
                            'poll_frequency',
                            'How often to query the database for the key (ms)',
                            intent='in')
    num_tries = fortran_argument('integer',
                    'num_tries',
                    'How many times to query the database before failing',
                    intent='in')
    success = fortran_argument('logical',
                    'success',
                    'True if the key was found AND matched the requested value')
    # C-equivalent types
    c_poll_frequency = fortran_argument('integer',
                    'poll_frequency',
                    'How often to query the database for the key (ms)',
                    kind='c_int',
                    attributes = ['value'],
                    intent='in')
    c_num_tries = fortran_argument('integer',
                    'num_tries',
                    'How many times to query the database before failing',
                    kind='c_int',
                    attributes = ['value'],
                    intent='in')
    c_success = fortran_argument('logical',
                    'success',
                    'True if the key was found AND matched the requested value',
                    kind='c_bool')

    # Define the main procedures for routines with key prefixing
    put_array = fortran_procedure('put_array',
        'Generic interface for putting an array into the database')
    get_array = fortran_procedure('get_array',
        'Generic interface for retrieving an array from the database')
    put_scalar = fortran_procedure('put_scalar',
        'Generic interface for putting a scalar into the database')
    # Note get_scalar cannot be overloaded since the input arguments in the
    # signature are the same regardless of variable type
    get_scalar = fortran_procedure('get_scalar',
        'Generic interface for retrieving a scalar from the database')
    poll_key_and_check = fortran_procedure('poll_key_and_check_scalar',
        'Generic interface for polling database for a key and checking'\
            ' its value')
    # Define the procedures for routines without key prefixing
    put_exact_key_array = fortran_procedure('put_exact_key_array',
        'Generic interface for putting an array into the database with '\
            'the exact key')
    get_exact_key_array = fortran_procedure('get_exact_key_array',
        'Generic interface for retrieving an array from the database with '\
            'the exact key')
    put_exact_key_scalar = fortran_procedure('put_exact_key_scalar',
        'Generic interface for putting a scalar into the database with the '\
            'exact key')
    get_exact_key_scalar = fortran_procedure('get_exact_key_scalar',
        'Generic interface for retrieving a scalar from the database with the '\
            'exact_key')
    poll_exact_key_and_check = fortran_procedure(
        'poll_exact_key_and_check_scalar',
        'Generic interface for polling database for an exact key and checking '\
            'its value')

    # Define all the variable types to create routines for
    var_types = {
        'int32': { 'c_kind':'c_int32_t', 'f_type':'integer', 'kind':4 },
        'int64': { 'c_kind':'c_int64_t', 'f_type':'integer', 'kind':8 },
        'float': { 'c_kind':'c_float',   'f_type':'real', 'kind':4 },
        'double':{ 'c_kind':'c_double',  'f_type':'real', 'kind':8 }
        }

    # Generate routines for each variable type
    for varname, var_d in var_types.items():
        # Define variables which have different types
        array_put = fortran_argument(var_d['f_type'],
                                     'array',
                                     'Array to send to database',
                                     attributes=['dimension(..)','target'],
                                     kind=var_d['kind'],
                                     intent='in')
        array_get = fortran_argument(var_d['f_type'],
                                     'array',
                                     'Array to be received from database',
                                     attributes=['dimension(..)','target'],
                                     kind=var_d['kind'],
                                     intent = 'inout')
        check_value = fortran_argument(var_d['f_type'],
                            'check_value',
                            'Value against which the key will be compared',
                            kind=var_d['kind'],
                            intent='in')
        scalar_put = fortran_argument(var_d['f_type'],
                                      'scalar',
                                      'Scalar value to send to database',
                                      kind=var_d['kind'],
                                      intent='in')
        scalar_get = fortran_argument(var_d['f_type'],
                                      'scalar',
                                      'Scalar value to get from the database',
                                      intent='out',
                                      kind=var_d['kind'])

        # Equivalent iso C arguments
        c_check_value = fortran_argument(var_d['f_type'],
                            'check_value',
                            'Value against which the key will be compared',
                            kind=var_d['c_kind'],
                            attributes = ['value'],
                            intent='in')
        c_scalar_put = fortran_argument(var_d['f_type'],
                                      'scalar',
                                      'Scalar value to send to database',
                                      kind=var_d['c_kind'],
                                      attributes = ['value'],
                                      intent='in')
        c_scalar_get = fortran_argument(var_d['f_type'],
                                      'scalar',
                                      'Scalar value to get from the database',
                                      kind=var_d['c_kind'])

        # Routines
        put_array.members.append( fortran_routine('subroutine',
                            f'put_array_{varname}',
                            f'Store an array of type {varname} in the database',
                            [ssc, f_key, array_put],
                            [ssc, c_key, array_ptr, dims, ndims]) )
        poll_key_and_check.members.append( fortran_routine('function',
                        f'poll_key_and_check_scalar_{varname}',
                        f'Check for existence of a key and whether it matches the requested value of type {varname}',
                        [ssc, f_key, check_value, poll_frequency, num_tries],
                        [ssc, c_key, c_check_value, c_poll_frequency, c_num_tries],
                        c_result = c_success,
                        result=success))
        get_array.members.append(fortran_routine('subroutine',
                        f'get_array_{varname}',
                        f'Get an array of type {varname} from the database',
                        [ssc, f_key, array_get],
                        [ssc, c_key, array_ptr, dims, ndims]))
        put_scalar.members.append(fortran_routine('subroutine',
                        f'put_scalar_{varname}',
                        f'Put a scalar of type {varname} to the database',
                        [ssc, f_key, scalar_put],
                        [ssc, c_key, c_scalar_put]))
        get_scalar.members.append(fortran_routine('subroutine',
                        f'get_scalar_{varname}',
                        f'Get a scalar of type {varname} from the database',
                        [ssc, f_key, scalar_get],
                        [ssc, c_key],
                        c_result = c_scalar_get))

        put_exact_key_array.members.append( fortran_routine('subroutine',
            f'put_exact_key_array_{varname}',
            f'Store an array of type {varname} in the database using exact key',
            [ssc, f_key, array_put],
            [ssc, c_key, array_ptr, dims, ndims]) )
        poll_exact_key_and_check.members.append( fortran_routine('function',
            f'poll_exact_key_and_check_scalar_{varname}',
            f'Check for existence of an exact key and whether it matches the '\
                f'requested value of type {varname}',
            [ssc, f_key, check_value, poll_frequency, num_tries],
            [ssc, c_key, c_check_value, c_poll_frequency, c_num_tries],
            c_result=c_success,
            result=success))
        get_exact_key_array.members.append(fortran_routine('subroutine',
            f'get_exact_key_array_{varname}',
            f'Get an array of type {varname} from the database using the '\
                'exact key',
            [ssc, f_key, array_get],
            [ssc, c_key, array_ptr, dims, ndims]))
        put_exact_key_scalar.members.append(fortran_routine('subroutine',
            f'put_exact_key_scalar_{varname}',
            f'Put a scalar of type {varname} to the database using '\
                'the exact key',
            [ssc, f_key, scalar_put],
            [ssc, c_key, c_scalar_put]))
        get_exact_key_scalar.members.append(fortran_routine('subroutine',
                f'get_exact_key_scalar_{varname}',
                f'Get a scalar of type {varname} from the database using '\
                    'the exact key',
                [ssc, f_key, scalar_get],
                [ssc, c_key],
                c_result = c_scalar_get))

    # Write various parts of the procedures into different files.
    # These get included into fortran_client.F90
    all_procedures = [put_array, poll_key_and_check, get_array, put_scalar,
                      get_scalar, put_exact_key_array,
                      poll_exact_key_and_check, get_exact_key_array,
                      put_exact_key_scalar, get_exact_key_scalar ]

    # Define the output files:
    header = '! This code was automatically generated\n'
    footer = '! End of generated code\n'

    with open(args.outdir + 'fortran_interface.inc','w') as interface_f:
        with open(args.outdir + 'fortran_routines.inc','w') as routine_f:
            print(header, file = interface_f)
            print(header, file = routine_f)
            print('!>\\file',file=routine_f)
            print('!>\\brief User-facing Fortran wrappers to SmartSim',
                  file=routine_f)
            for procedure in all_procedures:
                for routine in procedure:
                    routine.write_c_interface(file=interface_f)
                    routine.write_fortran_routine(file=routine_f)
            print(footer, file = interface_f)
            print(footer, file = routine_f)

    with open(args.outdir + 'fortran_header.inc','w') as header_f:
        print(header, file=header_f)
        print('!>\\file',file=header_f)
        print(
            '!>\\brief Defines user-facing interfaces for overloaded '\
                'SmartSim functions', file=header_f)
        for procedure in all_procedures:
            procedure.public_listing(file=header_f)
        for procedure in all_procedures:
            procedure.procedure_declaration(file=header_f)
        print(footer,file=header_f)