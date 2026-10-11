version_arr = meson.project_version().split('.')
version_config = configuration_data()
version_config.set('PL_MAJOR', version_arr[0])
version_config.set('PL_MINOR', version_arr[1])
version_config.set('PL_PATCH', version_arr[2])
version_config.set('PL_UPSTREAM_API', version_arr[1].to_int() / 10000)
version_config.set('PL_DOWNSTREAM_ABI', version_arr[1].to_int() % 10000)

libplacebo_rc = configure_file(input: 'libplacebo.rc.in',
                               output: 'libplacebo.rc',
                               configuration: version_config)
demos_rc =  configure_file(input: 'demos.rc.in',
                           output: 'demos.rc',
                           configuration: version_config)
