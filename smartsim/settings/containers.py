class Container():
    '''Base class for container types in SmartSim.

    Container types are used to embed all the information needed to
    launch a workload within a container into a single object.

    :param image: local or remote path to container image
    :type image: str
    :param args: number of cpus per node, defaults to None
    :type args: str | list[str], optional
    :param bind_paths: paths to bind (mount) from host machine into image.
    :type bind_paths: str | list[str] | dict[str, str], optional
    '''

    def __init__(self, image, args='', bind_paths=None):
        # Validate types
        if not isinstance(image, str):
            raise TypeError('image must be a str')
        elif not isinstance(args, (str, list)):
            raise TypeError('args must be a str | list')
        elif not isinstance(bind_paths, (str, list, dict)):
            raise TypeError('bind_paths must be a str | list | dict')

        self.image = image
        self.args = args
        self.bind_paths = bind_paths

    def _containerized_run_command(self, run_command: str):
        '''Return modified run_command with container commands prepended.

        :param run_command: run command from a RunSettings class
        :type run_command: str
        '''
        raise NotImplementedError(f"Containerized run command specification not implemented for this Container type: {type(self)}")


class Singularity(Container):
    '''Singularity container type.

    :param image: local or remote path to container image
    :type image: str
    :param args: number of cpus per node, defaults to None
    :type args: str | list[str], optional
    :param bind_paths: paths to bind (mount) from host machine into image.
    :type bind_paths: str | list[str] | dict[str, str], optional
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _containerized_run_command(self, run_command: str):
        '''Return modified run_command with container commands appended.

        :param run_command: run command from a Settings class
        :type run_command: str
        :raises TypeError: if object members are invalid types
        '''
        # Serialize args into a str
        if isinstance(self.args, str):
            serialized_args = self.args
        elif isinstance(self.args, list):
            serialized_args = ' '.join(self.args)
        else:
            raise TypeError('self.args must be a str | list')

        # Serialize bind_paths into a str
        if isinstance(self.bind_paths, str):
            serialized_bind_paths = self.bind_paths
        elif isinstance(self.bind_paths, list):
            serialized_bind_paths = ','.join(self.bind_paths)
        elif isinstance(self.bind_paths, dict):
            paths = []
            for host_path,img_path in self.bind_paths.items():
                if img_path:
                    paths.append(f'{host_path}={img_path}')
                else:
                    paths.append(host_path)
            serialized_bind_paths = ','.join(paths)
        else:
            raise TypeError('self.bind_paths must be str | list | dict')

        # Construct containerized launch command
        new_command = f'{run_command} singularity {self.image} {serialized_args} --bind {serialized_bind_paths}'
        return new_command


if __name__ == '__main__':

    #
    # args types
    #

    # args=str
    s = Singularity('image.sif', args='--nv', bind_paths='/foo/bar')
    print(s._containerized_run_command('srun -n 16'))
    # singularity image.sif --nv --bind /foo/bar srun -n 16 myapp.py --verbose
    # args=list(str)
    s = Singularity('image.sif', args=['--nv', '-v'], bind_paths='/foo/bar')
    print(s._containerized_run_command('srun -n 16'))
    # singularity image.sif --nv -v --bind /foo/bar srun -n 16 myapp.py --verbose

    #
    # bind_paths types
    #

    # bind_paths:str
    s = Singularity('image.sif', args='--nv', bind_paths='/foo/bar')
    print(s._containerized_run_command('srun -n 16'))
    # singularity image.sif --nv --bind /foo/bar srun -n 16 myapp.py --verbose

    # bind_paths:list(str)
    s = Singularity('image.sif', bind_paths=['/foo/bar', '/baz/'])
    print(s._containerized_run_command('srun -n 16'))
    # singularity image.sif  --bind /foo/bar,/baz/ srun -n 16 myapp.py --verbose

    # bind_paths:dict(str,str)
    s = Singularity('image.sif', bind_paths={'/foo/bar':'/foo/baz', '/a/b/c':'/usr/opt/c', '/baz/': None})
    print(s._containerized_run_command('srun -n 16'))
    # singularity image.sif  --bind /foo/bar=/foo/baz,/a/b/c=/usr/opt/c,/baz/ srun -n 16 myapp.py --verbose

