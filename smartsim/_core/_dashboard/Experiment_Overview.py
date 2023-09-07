import pandas as pd
import streamlit as st
import typing as t
from smartsim._core._dashboard.utils.pageSetup import (
    local_css,
    set_streamlit_page_config,
)
from smartsim._core._dashboard.utils.FileReader import ManifestReader

set_streamlit_page_config()
local_css("smartsim/_core/_dashboard/assets/style.scss")

# get real path and manifest.json
manifest_data = ManifestReader("tests/test_dashboard/manifest_files/manifesttest.json")

if manifest_data.data is None or manifest_data.experiment == {}:
    st.header("Experiment Not Found")
else:
    st.header(
        "Experiment Overview: "
        + manifest_data.get_entity_value("name", manifest_data.experiment)
    )

st.write("")

experiment, application, orchestrators, ensembles = st.tabs(
    ["Experiment", "Applications", "Orchestrators", "Ensembles"]
)

### Experiment ###
with experiment:
    st.subheader("Experiment Configuration")
    st.write("")
    col1, col2 = st.columns([4, 4])
    with col1:
        st.write(
            "Status: :green[Running]"
        )
        st.write(
            "Path: " + manifest_data.get_entity_value("path", manifest_data.experiment)
        )
        st.write(
            "Launcher: "
            + manifest_data.get_entity_value("launcher", manifest_data.experiment)
        )

    st.write("")
    with st.expander(label="Logs"):
        col1, col2 = st.columns([6, 6])
        with col1:
            st.write("Output")
            st.info("")

        with col2:
            st.write("Error")
            st.info("")

### Applications ###
with application:
    st.subheader("Application Configuration")
    col1, col2 = st.columns([4, 4])
    with col1:
        selected_app_name: t.Optional[str] = st.selectbox(
            "Select an application:",
            [app["name"] for app in manifest_data.applications],
        )
    if selected_app_name is not None:
        SELECTED_APPLICATION = manifest_data.get_entity(
            selected_app_name, manifest_data.applications
        )
    else:
        SELECTED_APPLICATION = None

    st.write("")
    st.write(
        "Status: :green[Running]"
    )
    st.write("Path: " + manifest_data.get_entity_value("path", SELECTED_APPLICATION))

    st.write("")
    with st.expander(label="Executable Arguments"):
        exec_arg_names = manifest_data.get_entity_value(
            "exe_args", SELECTED_APPLICATION
        )
        exec_args = {
            "All Arguments": exec_arg_names,
        }
        df = pd.DataFrame(exec_args)
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.write("")
    with st.expander(label="Batch and Run Settings"):
        col1, col2 = st.columns([4, 4])
        batch_names, batch_values = manifest_data.get_entity_dict_keys_and_values(
            "batch_settings", SELECTED_APPLICATION
        )
        with col1:
            batch = {"Name": batch_names, "Value": batch_values}
            df = pd.DataFrame(batch)
            st.write("Batch")
            st.dataframe(df, hide_index=True, use_container_width=True)
        with col2:
            run_names, run_values = manifest_data.get_entity_dict_keys_and_values(
                "run_settings", SELECTED_APPLICATION
            )
            rs = {"Name": run_names, "Value": run_values}
            df = pd.DataFrame(rs)
            st.write("Run")
            st.dataframe(df, hide_index=True, use_container_width=True)

    st.write("")
    with st.expander(label="Parameters and Generator Files"):
        col1, col2 = st.columns([4, 4])
        with col1:
            (
                param_names,
                param_values,
            ) = manifest_data.get_entity_dict_keys_and_values(
                "params", SELECTED_APPLICATION
            )
            params = {
                "Name": param_names,
                "Value": param_values,
            }
            df = pd.DataFrame(params)
            st.write("Parameters")
            st.dataframe(df, hide_index=True, use_container_width=True)
        with col2:
            file_type, file_paths = manifest_data.get_entity_dict_keys_and_values(
                "files", SELECTED_APPLICATION
            )
            files = {
                "File": file_paths,
                "Type": file_type,
            }
            df = pd.DataFrame(files)
            st.write("Files")
            st.dataframe(df, hide_index=True, use_container_width=True)

    st.write("")
    with st.expander(label="Colocated Database"):
        with st.container():
            col1, col2 = st.columns([6, 6])
            with col1:
                st.write("Summary")
                colo_keys, colo_vals = manifest_data.get_entity_dict_keys_and_values(
                    "colocated_db_settings", SELECTED_APPLICATION
                )
                colo_db = {"Name": colo_keys, "Value": colo_vals}
                df = pd.DataFrame(colo_db)
                st.dataframe(df, hide_index=True, use_container_width=True)

            with col2:
                st.write("Loaded Entities")
                entities = manifest_data.get_loaded_entities(SELECTED_APPLICATION)
                df = pd.DataFrame(entities)
                st.dataframe(df, hide_index=True, use_container_width=True)

    st.write("")
    with st.expander(label="Logs"):
        with st.container():
            col1, col2 = st.columns([6, 6])
            with col1:
                st.write("Output")
                st.info("")

            with col2:
                st.write("Error")
                st.info("")

### Orchestrator ###
with orchestrators:
    st.subheader("Orchestrator Configuration")
    col1, col2 = st.columns([4, 4])
    with col1:
        selected_orc_name: t.Optional[str] = st.selectbox(
            "Select an orchestrator:",
            [orc["name"] for orc in manifest_data.orchestrators],
        )

    if selected_orc_name is not None:
        SELECTED_ORC = manifest_data.get_entity(
            selected_orc_name, manifest_data.orchestrators
        )
    else:
        SELECTED_ORC = None

    st.write("")
    st.write("Status: :green[Running]")
    st.write("Launcher: " + manifest_data.get_entity_value("launcher", SELECTED_ORC))
    st.write("Port: " + str(manifest_data.get_entity_value("port", SELECTED_ORC)))
    st.write("Interface: " + manifest_data.get_entity_value("interface", SELECTED_ORC))

    st.write("")
    with st.expander(label="Database Hosts"):
        hosts = {
            "Hosts": manifest_data.get_entity_value("db_hosts", SELECTED_ORC),
        }
        df = pd.DataFrame(hosts)
        st.dataframe(df, hide_index=True, use_container_width=True)
    st.write("")
    with st.expander(label="Logs"):
        col1, col2 = st.columns([6, 6])
        with col1:
            st.session_state["shard_name"] = st.selectbox(
                label="Shard", options=("Shard 1", "Shard 2", "Shard 3", "Shard 4")
            )
            st.write("")
            st.write("Output")
            out = st.empty()
            st.info("")

        with col2:
            st.write("#")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("Error")
            err = st.empty()
            st.info("")

### Ensembles ###
with ensembles:
    st.subheader("Ensemble Configuration")
    col1, col2 = st.columns([4, 4])
    with col1:
        selected_ensemble_name: t.Optional[str] = st.selectbox(
            "Select an ensemble:",
            [ensemble["name"] for ensemble in manifest_data.ensembles],
        )

    if selected_ensemble_name is not None:
        SELECTED_ENSEMBLE = manifest_data.get_entity(
            selected_ensemble_name, manifest_data.ensembles
        )
    else:
        SELECTED_ENSEMBLE = None

    st.write("")
    st.write("Status: :green[Running]")
    st.write(
        "Strategy: " + manifest_data.get_entity_value("perm_strat", SELECTED_ENSEMBLE)
    )

    st.write("")
    with st.expander(label="Batch Settings"):
        batch_names, batch_values = manifest_data.get_entity_dict_keys_and_values(
            "batch_settings", SELECTED_ENSEMBLE
        )
        batch = {
            "Name": batch_names,
            "Value": batch_values,
        }
        df = pd.DataFrame(batch)
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.write("")
    with st.expander(label="Parameters"):
        (
            ens_param_names,
            ens_param_values,
        ) = manifest_data.get_entity_dict_keys_and_values("params", SELECTED_ENSEMBLE)
        ens_params = {
            "Name": ens_param_names,
            "Value": ens_param_values,
        }
        df = pd.DataFrame(ens_params)
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.write("#")
    if selected_ensemble_name is not None:
        st.subheader(selected_ensemble_name + " Member Configuration")
    else:
        st.subheader("Member Configuration")
    col1, col2 = st.columns([4, 4])
    with col1:
        members = manifest_data.get_ensemble_members(SELECTED_ENSEMBLE)
        selected_member_name: t.Optional[str] = st.selectbox(
            "Select a member:",
            [member["name"] for member in members if member is not None],
        )

    if selected_member_name is not None:
        SELECTED_MEMBER = manifest_data.get_member(
            selected_member_name, SELECTED_ENSEMBLE
        )
    else:
        SELECTED_MEMBER = None

    st.write("")
    st.write("Status: :green[Running]")
    st.write("Path: " + manifest_data.get_entity_value("path", SELECTED_MEMBER))
    st.write("")
    with st.expander(label="Executable Arguments"):
        args = manifest_data.get_entity_value("exe_args", SELECTED_MEMBER)
        exec_args = {"All Arguments": args}
        df = pd.DataFrame(exec_args)
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.write("")
    with st.expander(label="Batch and Run Settings"):
        col1, col2 = st.columns([4, 4])
        with col1:
            (
                mem_batch_name,
                mem_batch_value,
            ) = manifest_data.get_entity_dict_keys_and_values(
                "batch_settings", SELECTED_MEMBER
            )
            batch = {
                "Name": mem_batch_name,
                "Value": mem_batch_value,
            }
            df = pd.DataFrame(batch)
            st.write("Batch")
            st.dataframe(df, hide_index=True, use_container_width=True)
        with col2:
            mem_rs_name, mem_rs_value = manifest_data.get_entity_dict_keys_and_values(
                "run_settings", SELECTED_MEMBER
            )
            rs = {
                "Name": mem_rs_name,
                "Value": mem_rs_value,
            }
            df = pd.DataFrame(rs)
            st.write("Run")
            st.dataframe(df, hide_index=True, use_container_width=True)

    st.write("")
    with st.expander(label="Parameters and Generator Files"):
        col1, col2 = st.columns([4, 4])
        with col1:
            (
                mem_param_name,
                mem_param_value,
            ) = manifest_data.get_entity_dict_keys_and_values("params", SELECTED_MEMBER)
            params = {
                "Name": mem_param_name,
                "Value": mem_param_value,
            }
            df = pd.DataFrame(params)
            st.write("Parameters")
            st.dataframe(df, hide_index=True, use_container_width=True)
        with col2:
            mem_file_type, mem_files = manifest_data.get_entity_dict_keys_and_values(
                "files", SELECTED_MEMBER
            )
            files = {
                "File": mem_files,
                "Type": mem_file_type,
            }
            df = pd.DataFrame(files)
            st.write("Files")
            st.dataframe(df, hide_index=True, use_container_width=True)

    st.write("")
    with st.expander(label="Colocated Database"):
        with st.container():
            col1, col2 = st.columns([6, 6])
            with col1:
                st.write("Summary")
                (
                    mem_colo_keys,
                    mem_colo_vals,
                ) = manifest_data.get_entity_dict_keys_and_values(
                    "colocated_db_settings", SELECTED_MEMBER
                )
                mem_colo_db = {"Name": mem_colo_keys, "Value": mem_colo_vals}
                df = pd.DataFrame(mem_colo_db)
                st.dataframe(df, hide_index=True, use_container_width=True)

            with col2:
                st.write("Loaded Entities")
                mem_entities = manifest_data.get_loaded_entities(SELECTED_MEMBER)
                df = pd.DataFrame(mem_entities)
                st.dataframe(df, hide_index=True, use_container_width=True)

    st.write("")
    with st.expander(label="Logs"):
        col1, col2 = st.columns([6, 6])
        with col1:
            st.write("Output")
            st.info("")

        with col2:
            st.write("Error")
            st.info("")
