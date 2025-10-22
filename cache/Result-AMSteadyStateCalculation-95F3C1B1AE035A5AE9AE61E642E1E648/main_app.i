### MESH AND GEOMETRY ###
lz=0.002
nz=13
bias=1.0
ad_steps=5
### MATERIAL PARAMETERS ###
### These values will be retrieved from Schiel data ###
temp_liq=1726.778
temp_sol=1664.645
temp_evap=2988.258980305507
### PHYSICAL PARAMETERS ###
temp_inf=353.0
base_plate_temp=353.0
ambient_p=100000.0
htc=20.0
emissivity=0.8
### HEAT SOURCE MODEL ###
### Only one has to be true, based on user`s selection.
hs_type='ELLIPSOID'
gaussian_heat_source=false
double_ellipsoidal_heat_source=true
conical_heat_source=false
is_surface_heat_source=false
### LASER PARAMETERS ###
P=55.45454545454545
u_beam=1.3696969696969699
r_beam=8.967E-5
semi_axis_x_rear=1.0512999999999999E-4
semi_axis_y=3.2749999999999996E-5
semi_axis_z=1.524E-5
beam_shape=gaussian
r_beam_ring=100E-6
ring_radius=100E-6
ring_fraction=0
n_tophat=5
### POWDER PARAMETERS ###
absor_b=44.58
powder_thickness=3.9999999999999996E-5
powder_density=0.8
k_air=0.05784
elem_per_layer=1
### Analytical melt pool dimensions as calculated from Rosenthals equation ###
halfWidth=2.4098107729158634E-5
meltPoolLength=1.6521039012407652E-4
mesh_level=0
### NUMERICAL AND BOUNDARY CONDITIONS ###
enable_damping=false
damping_factor=0.0
enable_SUPG=true
radiative_bc=true
convective_bc=true
evaporation_bc=true
with_fluid_flow=false
with_lower_bounds=false
with_keyhole=false
top_surface=5
with_upper_bounds=true
lower_bound_value=${base_plate_temp}
### OUTPUT VARIABLES ###
### IF with_fluid_flow = false, show_variables = 'temperature' ###
show_variables='temperature'
r_beam2=${fparse r_beam * r_beam}
lx=0.0030503219193481145
ly=3.6147161593737953E-4
meshsize=1.6256E-4
meshsize_min=5.08E-6
min_meshsize_factor=1.0
ny=3
nx=19
xmin=-0.00213522534354368
xmax=${fparse xmin+lx}
ymin=0.0
ymax=${fparse ly}
zmin=-0.00204
zmax=0
### HARD CODED PARAMETERS
dhdT_delta_min=50
ellipsoidal_hs_factor=1
conical_hs_factor=1
front_factor=0.5
rear_factor=2
adapt_region_front=1.7934E-4
adapt_region_rear=3.1392402873191305E-4
adapt_region_width=6.549999999999999E-5
adapt_region_depth=4.819621545831727E-5
adaptivity_steps=2
adaptivity_marker_no_powder='combo'
use_gradient_marker=true
use_gradient_matprop_marker=false
dT_gradient_indicator=${fparse (temp_liq - temp_sol)/2}
gradient_indicator=TC_GradientElementIndicator
recompute_indicators=true
normalize_error=true
max_error_scaling=5
coarsen=0.02
midrange=0.5
refine=0.8
gradient_prop_coarsen=0.01
gradient_prop_refine=0.2
midrange_adaptivity_level=${fparse max(1, ad_steps - mesh_level - 3)}
increase_in_must_refine=1.3
temp_max=${fparse max(temp_inf, base_plate_temp)}
dTemp_liq_sol=${fparse min(100, (temp_liq-temp_sol)/2)}
lower_bound=${fparse temp_sol - dTemp_liq_sol + (base_plate_temp-296.15)/2}
skip_bound_on_failure=true
preconditioning_type='SMP'
solve_type='NEWTON'
petsc_options_iname='-ksp_type -pc_type -pc_factor_mat_solver_package -snes_type'
petsc_options_value='preonly    lu       superlu_dist                  vinewtonssls'
const_at_first_step=false
automatic_scaling=true
l_max_its=2000
l_tol=1e-9
nl_abs_tol=1e-8
nl_rel_tol=1e-4
nl_max_its=400
line_search='bt'
slow_converge_tolerance=1E-5
slow_converge_rel_tol_on=1E-3
conservative_model=-1
param_convection=4
param_dissipation=1
executioner_type=TC_TransientForSteady
dt_min=0.125
end_time=3
enable_fail_safe=1
refine_adaptivity_level=${fparse ad_steps+1}
improve_must_refine=true
perf_graph=false
execute_postprocessors_on='final'
exodus_output_execute_on='final'
bottom_ratio=10.0
[Problem]
type = TC_FEProblem
recompute_indicators = ${recompute_indicators}
slow_converge_tolerant = ${slow_converge_tolerance}
slow_converge_rel_tol_on = ${slow_converge_rel_tol_on}
[]

liquid_id = 0
solid_id = 1
powder_id = 2
block_ids = '0 1 2'
block_solid_liquid_ids = '0 1'

[Mesh]
[base_domains]
type = TC_KeyholeMesh
dim = 3
nx = ${nx}
ny = ${ny}
nz = ${nz}
xmin = ${xmin}
xmax = ${xmax}
ymin = ${ymin}
ymax = ${ymax}
zmin = ${zmin}
zmax = ${zmax}
bias_z = ${bias}
with_keyhole = ${with_keyhole}
r_beam = ${r_beam}
absorptivity = ${fparse absor_b/100}
bottom_ratio = ${bottom_ratio}
[]
[]

[AuxVariables]
[bounds]
[]
[]

[Bounds]
[lower_bounding]
type = ConstantBounds
variable = bounds
bounded_variable = temperature
bound_type = lower
bound_value = ${lower_bound_value}
execute_on = 'LINEAR TIMESTEP_END'
enable = ${with_lower_bounds}
[]
[]

[AMHeatConductionAction]
model = STEADY_HEAT_ONLY
hs_type = ${hs_type}
power = ${P}
absorptivity = ${fparse absor_b/100}
beam_radius = ${r_beam}
u_beam = ${u_beam}
beam_shape = ${beam_shape}
beam_radius_ring = ${r_beam_ring}
ring_radius = ${ring_radius}
ring_fraction = ${ring_fraction}
n_tophat = ${n_tophat}
semi_axis_x_front = ${r_beam}
semi_axis_x_rear = ${semi_axis_x_rear}
semi_axis_y = ${semi_axis_y}
semi_axis_z = ${semi_axis_z}
radiative_bc = ${radiative_bc}
convective_bc =  ${convective_bc}
evaporation_bc =  ${evaporation_bc}
radiative_bc_sides = ${top_surface}
convective_bc_sides =  ${top_surface}
evaporation_bc_sides =  ${top_surface}
T_base = ${base_plate_temp}
T_infinity = ${temp_inf}
T_sol = ${temp_sol}
T_liq = ${temp_liq}
T_eva = ${temp_evap}
htc = ${htc}
emissivity_function = ${emissivity}
chamber_pressure = ${ambient_p}
delta_min = ${dhdT_delta_min}
with_keyhole = ${with_keyhole}
conservative_model = ${conservative_model}
param_convection = ${param_convection}
param_dissipation = ${param_dissipation}
enable_fail_safe = ${enable_fail_safe}
end_time = ${end_time}
[]

[Adaptivity]
initial_marker = initial_marker
initial_steps = ${ad_steps}
max_h_level = ${ad_steps}
cycles_per_step = ${ad_steps}
recompute_markers_during_cycles = true

marker = ${adaptivity_marker_no_powder}
steps = ${adaptivity_steps}

[Indicators]
[gradient_error]
type = ${gradient_indicator}
variable = temperature
enable = ${use_gradient_marker}
[]
[gradient_matprop_error]
type = TC_GradientPropertyIndicator
variable = temperature
mat_prop = dHdT
temp_sol = ${temp_sol}
temp_liq = ${temp_liq}
dT_gradient = ${dT_gradient_indicator}
enable = ${use_gradient_matprop_marker}
[]
[]

[Markers]
[initial_marker]
type = WeldPoolMarker
lower_bound = ${lower_bound}
upper_bound = 10000
variable = temperature
max_h_level_on_top_surface = ${ad_steps}
max_h_level_on_mushy_zone = ${ad_steps}
max_h_level_on_liquid_zone = ${ad_steps}
must_refine = 'sqrt((x>0)*x^2/${adapt_region_front}^2 + (x<0)*x^2/${adapt_region_rear}^2 + y^2/${adapt_region_width}^2 + z^2/${adapt_region_depth}^2)'
beam_radius = ${r_beam}
mesh_size_factor = ${min_meshsize_factor}
[]
[markers_norefine]
type = WeldPoolMarker
lower_bound = ${lower_bound}
upper_bound = 10000
variable = temperature
max_h_level_on_top_surface = ${ad_steps}
max_h_level_on_mushy_zone = ${ad_steps}
max_h_level_on_liquid_zone = ${ad_steps}
beam_radius = ${r_beam}
mesh_size_factor = ${min_meshsize_factor}
skip_bound_on_failure = ${skip_bound_on_failure}
[]
[markers_mustrefine]
type = WeldPoolMarker
lower_bound = ${lower_bound}
upper_bound = 10000
variable = temperature
max_h_level_on_top_surface = ${ad_steps}
max_h_level_on_mushy_zone = ${ad_steps}
max_h_level_on_liquid_zone = ${ad_steps}
must_refine = 'sqrt((x>0)*x^2/${adapt_region_front}^2 + (x<0)*x^2/${adapt_region_rear}^2 + y^2/${adapt_region_width}^2 + z^2/${adapt_region_depth}^2)'
beam_radius = ${r_beam}
mesh_size_factor = ${min_meshsize_factor}
improve_must_refine = ${improve_must_refine}
[]
[gradient_marker]
type = TC_ErrorToleranceMarker
indicator = gradient_error
coarsen = ${coarsen}
midrange = ${midrange}
refine = ${refine}
midrange_adaptivity_level = ${midrange_adaptivity_level}
beam_radius = ${r_beam}
min_meshsize_factor = ${min_meshsize_factor}
normalize_error = ${normalize_error}
max_error_scaling = ${max_error_scaling}
enable = ${use_gradient_marker}
[]
[gradient_matprop_marker]
type = TC_ErrorToleranceMarker
indicator = gradient_matprop_error
coarsen = ${gradient_prop_coarsen}
refine = ${gradient_prop_refine}
beam_radius = ${r_beam}
min_meshsize_factor = ${min_meshsize_factor}
enable = ${use_gradient_matprop_marker}
[]
[combo]
type = TC_ComboMarker
markers = 'markers_norefine gradient_marker gradient_matprop_marker'
kh_markers = 'markers_norefine gradient_marker gradient_matprop_marker'
failure_markers = 'markers_mustrefine'
outputs = none
[]
[]
[]

[Preconditioning]
[SMP_n]
type = ${preconditioning_type}
solve_type = ${solve_type}
petsc_options_iname = ${petsc_options_iname}
petsc_options_value = ${petsc_options_value}
[]
[]

[Dampers]
# Use a constant damping parameter
[diffusion_damp]
type = ConstantDamper
damping = ${damping_factor}
enable = ${enable_damping}
[]
[]

[Executioner]
type = ${executioner_type}
end_time = ${end_time}
dt = 1
dtmin = ${dt_min}
refine_adaptivity_level = ${refine_adaptivity_level}

automatic_scaling = ${automatic_scaling}
l_max_its = ${l_max_its}
l_tol = ${l_tol}
nl_abs_tol = ${nl_abs_tol}
nl_rel_tol = ${nl_rel_tol}
nl_max_its = ${nl_max_its}
line_search = ${line_search}
increase_in_must_refine = ${increase_in_must_refine}
[]

[Postprocessors]

[size_x]
type = TC_IsoContourSize
contour_pos = iso_l_positions
comp = x
execute_on = ${execute_postprocessors_on}
[]
[size_y]
type = TC_IsoContourSize
contour_pos = iso_l_positions
comp = y
execute_on = ${execute_postprocessors_on}
[]
[size_z]
type = TC_IsoContourSize
contour_pos = iso_l_positions
comp = z
execute_on = ${execute_postprocessors_on}
[]
[adaptivity_level]
type = TC_AdaptivityLevel
size_y = size_y
size_z = size_z
meshsize = ${meshsize}
mesh_level = ${mesh_level}
is_surface_heat_source = ${is_surface_heat_source}
execute_on = 'timestep_begin timestep_end'
[]
[max_top_temp]
type = SideExtremeValue
variable = temperature
boundary = ${top_surface}
execute_on = 'timestep_end'
[]
[]

[VectorPostprocessors]
[melted_pool]
type = TC_IsoValueBoundingBox
iso_uo = iso_l_positions
outputs = csv
execute_on = final
[]
[haz]
type = TC_IsoValueBoundingBox
iso_uo = iso_s_positions
outputs = csv
execute_on = final
[]
[]

[UserObjects]
[iso_s_positions]
type = TC_IsoValueFinder
variable = temperature
value = ${temp_sol}
execute_on = 'final'
[]
[iso_l_positions]
type = TC_IsoValueFinder
variable = temperature
value = ${temp_liq}
execute_on = 'initial timestep_begin timestep_end final'
[]
[upper_bounding]
type = TC_ConstantBound
variable = temperature
max_top_temp = max_top_temp
upper_bound = ${temp_evap}
with_keyhole = ${with_keyhole}
enable = ${with_upper_bounds}
execute_on = 'timestep_end'
[]
[]

[Outputs]
console = false
perf_graph = ${perf_graph}
wall_time_checkpoint = false
[console]
type = TC_Console
execute_postprocessors_on = none
execute_on = NONLINEAR
[]
[csv]
type = CSV
execute_vector_postprocessors_on = final
hide = 'max_domain_temp max_top_temp'
file_base="/var/folders/zp/tn282fks34s0lrxskwmkzvz40000gs/T/TC_TMP17144981170608886376/AMSteadyStateCalculation-c24146d2-804e-4a71-ad6f-6f0be6875ede/pp"
[]
[pp_temp]
type = CSV
execute_postprocessors_on = ${execute_postprocessors_on}
execute_vector_postprocessors_on = 'none'
file_base="/var/folders/zp/tn282fks34s0lrxskwmkzvz40000gs/T/TC_TMP17144981170608886376/AMSteadyStateCalculation-c24146d2-804e-4a71-ad6f-6f0be6875ede/pp_temp_0001"
hide = 'size_x size_y size_z adaptivity_level max_top_temp'
time_column = false
[]
[exodus]
type=TC_VTKOutput
file_base="/var/folders/zp/tn282fks34s0lrxskwmkzvz40000gs/T/TC_TMP17144981170608886376/AMSteadyStateCalculation-c24146d2-804e-4a71-ad6f-6f0be6875ede/result"
output_material_properties = true
hide = 'gradient_marker combo initial_marker markers_norefine markers_mustrefine size_x size_y size_z dk/dtemperature dmolar_volume/dtemperature bounds'
execute_on = ${exodus_output_execute_on}
execute_input_on = none
execute_postprocessors_on = none
# show = 'temperature k gamma liquid_vfrac'
[]
[]
