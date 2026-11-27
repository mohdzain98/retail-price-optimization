# Keyword-based Execution Paths

These diagrams show execution flows related to key features, auto-detected from the project structure.

## Feature: `data`

### Representative Paths

#### Path Group 1

```mermaid
flowchart LR
  scripts_check_data_py["scripts/check_data.py"]
  src_prep___init___py["src/prep/__init__.py"]
  src_prep_features_py["src/prep/features.py"]
  src_prep_loader_py["src/prep/loader.py"]
  src_prep_panel_py["src/prep/panel.py"]
  src_prep_service_py["src/prep/service.py"]
  src_prep_trend_py["src/prep/trend.py"]
  scripts_check_data_py --> src_prep___init___py
  src_prep___init___py --> src_prep_loader_py
  src_prep___init___py --> src_prep_service_py
  src_prep_service_py --> src_prep_features_py
  src_prep_service_py --> src_prep_panel_py
  src_prep_service_py --> src_prep_trend_py
```
**Execution chain:** scripts/check_data.py → src/prep/__init__.py → src/prep/service.py → src/prep/features.py

#### Path Group 2

```mermaid
flowchart LR
  scripts_check_data_py["scripts/check_data.py"]
  src_config_py["src/config.py"]
  src_schema_py["src/schema.py"]
  src_utils_exceptions_py["src/utils/exceptions.py"]
  scripts_check_data_py --> src_config_py
  src_config_py --> src_schema_py
  src_config_py --> src_utils_exceptions_py
```
**Execution chain:** scripts/check_data.py → src/config.py → src/schema.py

### Combined Flow (All `data` Paths)

```mermaid
flowchart LR
  scripts_check_data_py["scripts/check_data.py"]
  src_config_py["src/config.py"]
  src_prep___init___py["src/prep/__init__.py"]
  src_prep_features_py["src/prep/features.py"]
  src_prep_loader_py["src/prep/loader.py"]
  src_prep_panel_py["src/prep/panel.py"]
  src_prep_service_py["src/prep/service.py"]
  src_prep_trend_py["src/prep/trend.py"]
  src_schema_py["src/schema.py"]
  src_utils___init___py["src/utils/__init__.py"]
  src_utils_exceptions_py["src/utils/exceptions.py"]
  src_utils_logger_py["src/utils/logger.py"]
  src_utils_registry_py["src/utils/registry.py"]
  scripts_check_data_py --> src_config_py
  scripts_check_data_py --> src_prep___init___py
  scripts_check_data_py --> src_utils___init___py
  src_config_py --> src_schema_py
  src_config_py --> src_utils_exceptions_py
  src_prep___init___py --> src_prep_loader_py
  src_prep___init___py --> src_prep_service_py
  src_prep_service_py --> src_prep_features_py
  src_prep_service_py --> src_prep_panel_py
  src_prep_service_py --> src_prep_trend_py
  src_utils___init___py --> src_utils_logger_py
  src_utils___init___py --> src_utils_registry_py
```

## Feature: `scripts`

### Representative Paths

#### Path Group 1

```mermaid
flowchart LR
  scripts_check_data_py["scripts/check_data.py"]
  src_prep___init___py["src/prep/__init__.py"]
  src_prep_features_py["src/prep/features.py"]
  src_prep_loader_py["src/prep/loader.py"]
  src_prep_panel_py["src/prep/panel.py"]
  src_prep_service_py["src/prep/service.py"]
  src_prep_trend_py["src/prep/trend.py"]
  scripts_check_data_py --> src_prep___init___py
  src_prep___init___py --> src_prep_loader_py
  src_prep___init___py --> src_prep_service_py
  src_prep_service_py --> src_prep_features_py
  src_prep_service_py --> src_prep_panel_py
  src_prep_service_py --> src_prep_trend_py
```
**Execution chain:** scripts/check_data.py → src/prep/__init__.py → src/prep/service.py → src/prep/features.py

#### Path Group 2

```mermaid
flowchart LR
  scripts_create_user_py["scripts/create_user.py"]
  src_config_py["src/config.py"]
  src_schema_py["src/schema.py"]
  src_simulator_accounts_py["src/simulator/accounts.py"]
  scripts_create_user_py --> src_simulator_accounts_py
  src_config_py --> src_schema_py
  src_simulator_accounts_py --> src_config_py
```
**Execution chain:** scripts/create_user.py → src/simulator/accounts.py → src/config.py → src/schema.py

### Combined Flow (All `scripts` Paths)

```mermaid
flowchart LR
  scripts_check_data_py["scripts/check_data.py"]
  scripts_create_user_py["scripts/create_user.py"]
  src_config_py["src/config.py"]
  src_prep___init___py["src/prep/__init__.py"]
  src_prep_features_py["src/prep/features.py"]
  src_prep_loader_py["src/prep/loader.py"]
  src_prep_panel_py["src/prep/panel.py"]
  src_prep_service_py["src/prep/service.py"]
  src_prep_trend_py["src/prep/trend.py"]
  src_schema_py["src/schema.py"]
  src_simulator_accounts_py["src/simulator/accounts.py"]
  src_utils___init___py["src/utils/__init__.py"]
  src_utils_exceptions_py["src/utils/exceptions.py"]
  src_utils_logger_py["src/utils/logger.py"]
  src_utils_registry_py["src/utils/registry.py"]
  scripts_check_data_py --> src_config_py
  scripts_check_data_py --> src_prep___init___py
  scripts_check_data_py --> src_utils___init___py
  scripts_create_user_py --> src_simulator_accounts_py
  scripts_create_user_py --> src_utils___init___py
  src_config_py --> src_schema_py
  src_config_py --> src_utils_exceptions_py
  src_prep___init___py --> src_prep_loader_py
  src_prep___init___py --> src_prep_service_py
  src_prep_service_py --> src_prep_features_py
  src_prep_service_py --> src_prep_panel_py
  src_prep_service_py --> src_prep_trend_py
  src_simulator_accounts_py --> src_config_py
  src_utils___init___py --> src_utils_exceptions_py
  src_utils___init___py --> src_utils_logger_py
  src_utils___init___py --> src_utils_registry_py
```

## Feature: `user`

### Representative Paths

#### Path Group 1

```mermaid
flowchart LR
  scripts_create_user_py["scripts/create_user.py"]
  src_config_py["src/config.py"]
  src_schema_py["src/schema.py"]
  src_simulator_accounts_py["src/simulator/accounts.py"]
  scripts_create_user_py --> src_simulator_accounts_py
  src_config_py --> src_schema_py
  src_simulator_accounts_py --> src_config_py
```
**Execution chain:** scripts/create_user.py → src/simulator/accounts.py → src/config.py → src/schema.py

#### Path Group 2

```mermaid
flowchart LR
  scripts_create_user_py["scripts/create_user.py"]
  src_utils___init___py["src/utils/__init__.py"]
  src_utils_exceptions_py["src/utils/exceptions.py"]
  src_utils_logger_py["src/utils/logger.py"]
  src_utils_registry_py["src/utils/registry.py"]
  scripts_create_user_py --> src_utils___init___py
  src_utils___init___py --> src_utils_exceptions_py
  src_utils___init___py --> src_utils_logger_py
  src_utils___init___py --> src_utils_registry_py
```
**Execution chain:** scripts/create_user.py → src/utils/__init__.py → src/utils/exceptions.py

### Combined Flow (All `user` Paths)

```mermaid
flowchart LR
  scripts_create_user_py["scripts/create_user.py"]
  src_config_py["src/config.py"]
  src_schema_py["src/schema.py"]
  src_simulator_accounts_py["src/simulator/accounts.py"]
  src_utils___init___py["src/utils/__init__.py"]
  src_utils_exceptions_py["src/utils/exceptions.py"]
  src_utils_logger_py["src/utils/logger.py"]
  src_utils_registry_py["src/utils/registry.py"]
  scripts_create_user_py --> src_simulator_accounts_py
  scripts_create_user_py --> src_utils___init___py
  src_config_py --> src_schema_py
  src_simulator_accounts_py --> src_config_py
  src_utils___init___py --> src_utils_exceptions_py
  src_utils___init___py --> src_utils_logger_py
  src_utils___init___py --> src_utils_registry_py
```

---

<sub>Generated by [GNOST](https://gnost.readthedocs.io) 0.7.1. Try it on your own code: `uvx gnost onboard .`</sub>
