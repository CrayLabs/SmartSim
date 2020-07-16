# Define indents here to be two spaces
from argparse import ArgumentParser
indent = '  '

header_file = 'c_client.h'
source_file = 'c_client.cc'

class c_argument:
    """Contains all the descriptors for a C-style argument"""
    def __init__(self, var_type, name, description):
        """Initialize the descriptors for a C argument

        :param var_type: The type of the argument
        :type var_type: str
        :param name: The name of the argument
        :type name: str
        :param description: A brief description for what the variable is
        :type description: str
        """


        self.ctype = var_type
        self.generic_type = self.ctype
        if self.ctype.endswith('_t'):
            self.generic_type = var_type[:-2]
        self.name = name
        self.description = description
        if 'var_type' in self.ctype:
            self.prototype_declaration =  f'{self.ctype.format(var_type=var_type)} {self.name}'
        else:
            self.prototype_declaration =  f'{self.ctype} {self.name}'
        padspace = 24 - len(self.prototype_declaration)
        self.docstring =  self.prototype_declaration + ' '*padspace + f'/*!< {self.description} */'


class c_function:
    """Used to define and generate code for a C-style function
    """
    def __init__(self, func_type, name, description, input_args, cpp_args):
        """

        :param func_type: Return type of the function
        :type func_type: str
        :param name: Function name
        :type name: str
        :param description: A brief description of what the function does
        :type description: str
        :param input_args: A list of c_arguments used in the signature
        :type input_args: list
        :param cpp_args: Arguments needed for the call to the C++ method
        :type cpp_args: str
        """
        self.ctype = func_type
        self.cpp_name = name
        self.c_name = self.cpp_name + '_c'
        self.description = description
        self.input_args = input_args
        self.cpp_args = cpp_args

    def write_contents(self, var_type, indent='  ', file=None):
        """Generates the actual content of the function

        :param var_type: The variable type sent to/from the database
        :type var_type: str
        :param indent: the string used for indents, defaults to '  '
        :type indent: str, optional
        :param file: File to write the content to, defaults to None
        :type file: file handle, optional
        """

        generic_type = var_type
        if generic_type.endswith('_t'):
            generic_type = generic_type[:-2]
        contents = 'extern "C" '
        arglist = []
        for arg in self.input_args:
            if 'var_type' in arg.ctype:
                arglist.append(f'{var_type} {arg.name.format(var_type=var_type)}')
            else:
                arglist.append(f'{arg.ctype} {arg.name}')
        arg_declaration = ', '.join(arglist)
        if 'var_type' in self.ctype:
            contents += f'{self.ctype.format(var_type=var_type)} {self.c_name.format(var_type=generic_type)}({arg_declaration})\n'
        else:
            contents += f'{self.ctype} {self.c_name.format(var_type=generic_type)}({arg_declaration})\n'
        contents += '{\n'
        contents += indent+'SmartSimClient *s = (SmartSimClient *)SmartSimClient_p;\n'
        if 'void' in self.ctype:
            contents += indent+f's->{self.cpp_name.format(var_type=generic_type)}({self.cpp_args});\n'
        else:
            contents += indent+f'return s->{self.cpp_name.format(var_type=generic_type)}({self.cpp_args});\n'
        contents += '}\n'
        print(contents,file=file)

    def write_header(self, var_type, indent='  ', file=None):
        """Write the documented signature for the function

        :param var_type: Variable type to send/receive from the database
        :type var_type: str
        :param indent: string used for indents, defaults to '  '
        :type indent: str, optional
        :param file: File to write the header to, defaults to None
        :type file: [type], optional
        """
        generic_type = var_type
        if generic_type.endswith('_t'):
            generic_type = generic_type[:-2]
        contents =  f'//! {self.description.format(var_type=var_type)}\n'
        if 'var_type' in self.ctype:
            contents += f'{self.ctype.format(var_type=var_type)} {self.c_name.format(var_type=generic_type)}(\n'
        else:
            contents += f'{self.ctype} {self.c_name.format(var_type=generic_type)}(\n'
        arglist = []
        for arg in self.input_args:
            if 'var_type' in arg.docstring:
                arglist.append(indent*2 + arg.docstring.format(var_type=var_type))
            else:
                arglist.append(indent*2 + arg.docstring)
        contents += ',\n'.join(arglist)
        contents += '\n);\n'
        print(contents,file=file)

if __name__ == "__main__":
    # Parse the optional runtime arguments
    parser = ArgumentParser(
        description= "Generate C-wrappers for SmartSimClient methods")
    parser.add_argument('--outdir', default='../smartsim/clients/')
    args = parser.parse_args()

    # Define commonly used C arguments
    SmartSim_ptr = c_argument('void*', 'SmartSimClient_p',
        'Pointer to an initialized SmartSim Client')
    key = c_argument('const char*', 'key',
        'Identifier for this object in the database')
    array_put = c_argument('void*', 'array', 'Array to store in the database')
    array_get = c_argument('void*', 'array', 'Array to get from the database')
    scalar_put = c_argument('{var_type}', 'scalar',
        'Scalar value to store in the database')
    scalar_check = c_argument('{var_type}', 'scalar',
        'Scalar value against which to check')
    dimension = c_argument('int**', 'dimensions',
        'Length along each dimension of the array')
    ndims = c_argument('int*', 'ndims', 'Number of dimensions of the array')
    poll_frequency = c_argument('int', 'poll_frequency_ms',
        'How often to check the database in milliseconds')
    num_tries = c_argument('int', 'num_tries',
        'Number of times to check the database')

    # Define the SmartSim functions to be wrapped. Note this assumes that the
    # wrapper and the C++ class method has the same name
    put_array = c_function(
        'void',
        'put_array_{var_type}',
        'Put an array of type {var_type} into the database',
        [SmartSim_ptr, key, array_put, dimension, ndims],
        'key, array, *dimensions, *ndims')
    get_array = c_function(
        'void',
        'get_array_{var_type}',
        'Get an array of type {var_type} from the database',
        [SmartSim_ptr, key, array_get, dimension, ndims],
        'key, array, *dimensions, *ndims')
    put_scalar = c_function(
        'void',
        'put_scalar_{var_type}',
        'Put a scalar of type {var_type} into the database',
        [SmartSim_ptr, key, scalar_put],
        'key, scalar')
    get_scalar = c_function(
        '{var_type}',
        'get_scalar_{var_type}',
        'Get an array of type {var_type} from the database',
        [SmartSim_ptr, key],
        'key')
    poll_key_and_check_scalar = c_function(
        'int',
        'poll_key_and_check_scalar_{var_type}',
        'Poll the database for a key and check its value',
        [SmartSim_ptr, key, scalar_check, poll_frequency, num_tries],
        'key, scalar, poll_frequency_ms, num_tries')
    put_exact_key_array = c_function(
        'void',
        'put_exact_key_array_{var_type}',
        'Put an array of type {var_type} from the database using the exact key specified (no prefixing)',
        [SmartSim_ptr, key, array_put, dimension, ndims],
        'key, array, *dimensions, *ndims')
    get_exact_key_array = c_function(
        'void',
        'get_exact_key_array_{var_type}',
        'Get an array of type {var_type} from the database using the exact key specified (no prefixing)',
        [SmartSim_ptr, key, array_get, dimension, ndims],
        'key, array, *dimensions, *ndims')
    put_exact_key_scalar = c_function(
        'void',
        'put_exact_key_scalar_{var_type}',
        'Put an array of type {var_type} into the database using the exact key (no prefixing)',
        [SmartSim_ptr, key, scalar_put],
        'key, scalar')
    get_exact_key_scalar = c_function(
        '{var_type}',
        'get_exact_key_scalar_{var_type}',
        'Get an array of type {var_type} from the database using the exact key (no prefixing)',
        [SmartSim_ptr, key],
        'key')
    poll_exact_key_and_check_scalar = c_function(
        'int',
        'poll_exact_key_and_check_scalar_{var_type}',
        'Poll the database for an exact key and check its value',
        [SmartSim_ptr, key, scalar_check, poll_frequency, num_tries],
        'key, scalar, poll_frequency_ms, num_tries')

    # Create a list of the functions defined above to be generated
    functions = [ put_array, get_array, put_scalar, get_scalar,
                  poll_key_and_check_scalar,
                  put_exact_key_array, get_exact_key_array,
                  put_exact_key_scalar, get_exact_key_scalar,
                  poll_exact_key_and_check_scalar ]

    var_types = ['double','float','int64_t','int32_t','uint64_t','uint32_t']

    # Write the source code and header files for the C-interfaces
    with open(args.outdir + header_file,'w') as hfile:
        with open(args.outdir + source_file,'w') as sfile:

            # Write header lines for each file
            header = '/* The C wrappers in this file were autogenerated by c_code_generator.py */\n'

            # c_client.h header
            print(header,file=hfile)
            print('#ifndef SMARTSIM_C_CLIENT_H', file=hfile)
            print('#define SMARTSIM_C_CLIENT_H', file=hfile)
            # Following lines are for doxygen
            print('///@file',file=hfile)
            print('///\\brief C-wrappers for the C++ SmartSimClient class',file=hfile)
            print('#include "client.h"', file=hfile)
            print("#ifdef __cplusplus",file=hfile)
            print('extern "C" {',file=hfile)
            print("#endif",file=hfile)

            # c_client.c header
            print(header,file=sfile)
            print('#include "c_client.h"',file=sfile)

            for func in functions:
                for type in var_types:
                    func.write_header(type,file=hfile)
                    func.write_contents(type,file=sfile)

            # c_client.h footer
            print("#ifdef __cplusplus",file=hfile)
            print("}",file=hfile)
            print("#endif",file=hfile)
            print('#endif // SMARTSIM_C_CLIENT_H',file=hfile)

