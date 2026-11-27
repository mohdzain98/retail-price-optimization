# Full Execution Flow

```mermaid
flowchart TD
  dir_src_optimization["src/optimization<br/>8 files"]
  dir_src["src<br/>3 files"]
  dir_src_baseline["src/baseline<br/>5 files"]
  dir_scripts["scripts<br/>2 files"]
  dir_src_promo["src/promo<br/>8 files"]
  dir_src_utils["src/utils<br/>4 files"]
  dir_src_prep["src/prep<br/>6 files"]
  dir_src_elasticity["src/elasticity<br/>7 files"]
  dir_src_simulator["src/simulator<br/>1 file"]
  dir_scripts --> dir_src_prep
  dir_scripts --> dir_src_simulator
  dir_src --> dir_src_baseline
  dir_src --> dir_src_elasticity
  dir_src --> dir_src_optimization
  dir_src --> dir_src_prep
  dir_src --> dir_src_promo
  dir_src_baseline -->|2| dir_src_elasticity
  dir_src_promo -->|3| dir_src_elasticity
  dir_src_promo -->|2| dir_src_prep
```

> Grouped by folder to stay readable: 9 folders; numbers on arrows count the file-level imports they stand for.
> Arrows into `src`, `src/utils` are not drawn: most folders import them, and drawing every arrow would hide the structure.

# Detail by folder

The diagram above groups files by folder. These show the files inside the biggest folders.

## `src/optimization` (8 files)

```mermaid
flowchart TD
  src_optimization___init___py["src/optimization/__init__.py"]
  src_optimization_checks_py["src/optimization/checks.py"]
  src_optimization_demand_py["src/optimization/demand.py"]
  src_optimization_inputs_py["src/optimization/inputs.py"]
  src_optimization_problem_py["src/optimization/problem.py"]
  src_optimization_robustness_py["src/optimization/robustness.py"]
  src_optimization_service_py["src/optimization/service.py"]
  src_optimization_solver_py["src/optimization/solver.py"]
  src_optimization___init___py --> src_optimization_service_py
  src_optimization_checks_py --> src_optimization_problem_py
  src_optimization_problem_py --> src_optimization_demand_py
  src_optimization_problem_py --> src_optimization_inputs_py
  src_optimization_robustness_py --> src_optimization_demand_py
  src_optimization_robustness_py --> src_optimization_inputs_py
  src_optimization_robustness_py --> src_optimization_problem_py
  src_optimization_robustness_py --> src_optimization_solver_py
  src_optimization_service_py --> src_optimization_checks_py
  src_optimization_service_py --> src_optimization_demand_py
  src_optimization_service_py --> src_optimization_inputs_py
  src_optimization_service_py --> src_optimization_problem_py
  src_optimization_service_py --> src_optimization_robustness_py
  src_optimization_service_py --> src_optimization_solver_py
  src_optimization_solver_py --> src_optimization_problem_py
```

## `src/promo` (8 files)

```mermaid
flowchart TD
  src_promo___init___py["src/promo/__init__.py"]
  src_promo_candidates_py["src/promo/candidates.py"]
  src_promo_forecast_py["src/promo/forecast.py"]
  src_promo_forecasters_py["src/promo/forecasters.py"]
  src_promo_frontier_py["src/promo/frontier.py"]
  src_promo_kpis_py["src/promo/kpis.py"]
  src_promo_planner_py["src/promo/planner.py"]
  src_promo_service_py["src/promo/service.py"]
  src_promo___init___py --> src_promo_service_py
  src_promo_forecast_py --> src_promo_forecasters_py
  src_promo_frontier_py --> src_promo_planner_py
  src_promo_service_py --> src_promo_candidates_py
  src_promo_service_py --> src_promo_forecast_py
  src_promo_service_py --> src_promo_forecasters_py
  src_promo_service_py --> src_promo_frontier_py
  src_promo_service_py --> src_promo_kpis_py
  src_promo_service_py --> src_promo_planner_py
```

## `src/elasticity` (7 files)

```mermaid
flowchart TD
  src_elasticity___init___py["src/elasticity/__init__.py"]
  src_elasticity_base_py["src/elasticity/base.py"]
  src_elasticity_bayes_py["src/elasticity/bayes.py"]
  src_elasticity_frequentist_py["src/elasticity/frequentist.py"]
  src_elasticity_guardrails_py["src/elasticity/guardrails.py"]
  src_elasticity_service_py["src/elasticity/service.py"]
  src_elasticity_store_py["src/elasticity/store.py"]
  src_elasticity___init___py --> src_elasticity_base_py
  src_elasticity___init___py --> src_elasticity_bayes_py
  src_elasticity___init___py --> src_elasticity_frequentist_py
  src_elasticity___init___py --> src_elasticity_service_py
  src_elasticity___init___py --> src_elasticity_store_py
  src_elasticity_bayes_py --> src_elasticity_base_py
  src_elasticity_frequentist_py --> src_elasticity_base_py
  src_elasticity_guardrails_py --> src_elasticity_base_py
  src_elasticity_service_py --> src_elasticity_base_py
  src_elasticity_service_py --> src_elasticity_guardrails_py
  src_elasticity_service_py --> src_elasticity_store_py
  src_elasticity_store_py --> src_elasticity_base_py
```

## `src/prep` (6 files)

```mermaid
flowchart TD
  src_prep___init___py["src/prep/__init__.py"]
  src_prep_features_py["src/prep/features.py"]
  src_prep_loader_py["src/prep/loader.py"]
  src_prep_panel_py["src/prep/panel.py"]
  src_prep_service_py["src/prep/service.py"]
  src_prep_trend_py["src/prep/trend.py"]
  src_prep___init___py --> src_prep_loader_py
  src_prep___init___py --> src_prep_service_py
  src_prep_panel_py --> src_prep_loader_py
  src_prep_service_py --> src_prep_features_py
  src_prep_service_py --> src_prep_loader_py
  src_prep_service_py --> src_prep_panel_py
  src_prep_service_py --> src_prep_trend_py
```

## `src/baseline` (5 files)

```mermaid
flowchart TD
  src_baseline___init___py["src/baseline/__init__.py"]
  src_baseline_decomposition_py["src/baseline/decomposition.py"]
  src_baseline_effects_py["src/baseline/effects.py"]
  src_baseline_rules_py["src/baseline/rules.py"]
  src_baseline_service_py["src/baseline/service.py"]
  src_baseline___init___py --> src_baseline_service_py
  src_baseline_service_py --> src_baseline_decomposition_py
  src_baseline_service_py --> src_baseline_effects_py
  src_baseline_service_py --> src_baseline_rules_py
```

## `src/utils` (4 files)

```mermaid
flowchart TD
  src_utils___init___py["src/utils/__init__.py"]
  src_utils_exceptions_py["src/utils/exceptions.py"]
  src_utils_logger_py["src/utils/logger.py"]
  src_utils_registry_py["src/utils/registry.py"]
  src_utils___init___py --> src_utils_exceptions_py
  src_utils___init___py --> src_utils_logger_py
  src_utils___init___py --> src_utils_registry_py
  src_utils_registry_py --> src_utils_exceptions_py
```

## `src` (3 files)

```mermaid
flowchart TD
  src_config_py["src/config.py"]
  src_pipeline_py["src/pipeline.py"]
  src_schema_py["src/schema.py"]
  src_config_py --> src_schema_py
  src_pipeline_py --> src_config_py
```

> ℹ️ This file renders the complete Mermaid diagram on GitHub.
> For raw Mermaid source, see [**FLOW-full.mmd**](./FLOW-full.mmd). Every file and import, ungrouped and nothing omitted: [**flow-complete.mmd**](./flow-complete.mmd)

---

<sub>Generated by [GNOST](https://gnost.readthedocs.io) 0.7.1. Try it on your own code: `uvx gnost onboard .`</sub>
