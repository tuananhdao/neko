! Copyright (c) 2026, The Neko Authors
! All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:
!
!   * Redistributions of source code must retain the above copyright notice,
!     this list of conditions and the following disclaimer.
!   * Redistributions in binary form must reproduce the above copyright
!     notice, this list of conditions and the following disclaimer in the
!     documentation and/or other materials provided with the distribution.
!   * Neither the name of the authors nor the names of its contributors may be
!     used to endorse or promote products derived from this software without
!     specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
! AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.
!
!> CUDA implementation of the Euler GLL-IDP solver.
module euler_idp_device
  use, intrinsic :: ieee_arithmetic, only : ieee_is_finite
  use, intrinsic :: iso_c_binding, only : c_ptr, c_null_ptr
  use mpi_f08, only : MPI_Allreduce, MPI_MAX, MPI_MIN, MPI_SUM, MPI_INTEGER
  use num_types, only : rp
  use field, only : field_t
  use dofmap, only : dofmap_t
  use coefs, only : coef_t
  use gather_scatter, only : gs_t
  use gs_ops, only : GS_OP_ADD, GS_OP_MIN, GS_OP_MAX
  use bc_list, only : bc_list_t
  use bc, only : bc_t
  use time_state, only : time_state_t
  use device, only : device_map, device_unmap, device_memcpy, &
       HOST_TO_DEVICE, DEVICE_TO_HOST
  use device_math, only : device_add3s2, device_cmult2, device_copy, &
       device_glmax, device_glmin, device_glsum
  use euler_gll_graph, only : euler_gll_graph_t
  use euler_idp_backend, only : EULER_IDP_NCOMP, &
       EULER_IDP_DIAGNOSTICS_OFF, EULER_IDP_DIAGNOSTICS_FULL, &
       euler_idp_backend_t, euler_idp_diagnostics_t, &
       euler_idp_state_observation_t
#ifdef HAVE_CUDA
  use euler_idp_device_kernels
#endif
  use comm, only : NEKO_COMM, MPI_REAL_PRECISION
  use logger, only : LOG_SIZE
  use utils, only : neko_error
  implicit none
  private

  type, public, extends(euler_idp_backend_t) :: euler_idp_device_t
     logical :: periodic_graph = .false.
     logical :: low_order_only = .false.
     logical :: relax_density_bounds = .false.
     logical :: limit_internal_energy = .true.
     logical :: limit_entropy = .true.
     real(kind=rp) :: density_bound_relaxation_factor = 1.0_rp
     real(kind=rp) :: correction_tolerance = 1.0e-10_rp
     real(kind=rp) :: domain_volume = 0.0_rp
     real(kind=rp) :: density_relaxation_mass = 0.0_rp
     real(kind=rp) :: maximum_floor_timestep = huge(1.0_rp)
     real(kind=rp) :: limiter_weight_error = 0.0_rp
     integer :: time_order = 1
     type(euler_gll_graph_t) :: graph
     type(field_t) :: local_residual(EULER_IDP_NCOMP)
     type(field_t) :: low_candidate(EULER_IDP_NCOMP)
     type(field_t) :: saved_state(EULER_IDP_NCOMP)
     type(field_t) :: viscosity_sum
     type(field_t) :: density_lower_bound
     type(field_t) :: density_upper_bound
     type(field_t) :: entropy_lower_bound
     type(field_t) :: u, v, w, p, sound_speed, internal_energy
     type(field_t) :: work_1, work_2, work_3
     integer, allocatable :: edge_left(:), edge_right(:), edge_direction(:)
     real(kind=rp), allocatable :: edge_coefficient(:,:)
     real(kind=rp), allocatable :: diagonal_coefficient(:,:)
     real(kind=rp), allocatable :: edge_viscosity(:)
     real(kind=rp), allocatable :: correction_flux(:,:)
     real(kind=rp), allocatable :: edge_work(:)
     real(kind=rp), allocatable :: edge_limited(:)
     real(kind=rp), allocatable :: edge_density_limited(:)
     real(kind=rp), allocatable :: edge_energy_limited(:)
     real(kind=rp), allocatable :: edge_entropy_limited(:)
     real(kind=rp), allocatable :: directional_error(:)
     type(c_ptr) :: edge_left_d = c_null_ptr
     type(c_ptr) :: edge_right_d = c_null_ptr
     type(c_ptr) :: edge_direction_d = c_null_ptr
     type(c_ptr) :: edge_coefficient_d = c_null_ptr
     type(c_ptr) :: diagonal_coefficient_d = c_null_ptr
     type(c_ptr) :: edge_viscosity_d = c_null_ptr
     type(c_ptr) :: correction_flux_d = c_null_ptr
     type(c_ptr) :: edge_work_d = c_null_ptr
     type(c_ptr) :: edge_limited_d = c_null_ptr
     type(c_ptr) :: edge_density_limited_d = c_null_ptr
     type(c_ptr) :: edge_energy_limited_d = c_null_ptr
     type(c_ptr) :: edge_entropy_limited_d = c_null_ptr
     type(c_ptr) :: directional_error_d = c_null_ptr
   contains
     procedure, pass(this) :: init => euler_idp_device_init
     procedure, pass(this) :: init_graph => euler_idp_device_init_graph
     procedure, pass(this) :: free => euler_idp_device_free
     procedure, pass(this) :: apply_boundary_conditions => &
          euler_idp_device_apply_boundary_conditions
     procedure, pass(this) :: prepare_stage => euler_idp_device_prepare_stage
     procedure, pass(this) :: forward_euler => euler_idp_device_forward_euler
     procedure, pass(this) :: apply_candidate_boundary => &
          euler_idp_device_apply_candidate_boundary
     procedure, pass(this) :: observe_candidate => &
          euler_idp_device_observe_candidate
     procedure, pass(this) :: validate_candidate => &
          euler_idp_device_validate_candidate
     procedure, pass(this) :: save_state => euler_idp_device_save_state
     procedure, pass(this) :: combine_stage => euler_idp_device_combine_stage
     procedure, pass(this) :: copy_primitives => &
          euler_idp_device_copy_primitives
  end type euler_idp_device_t

contains

  subroutine euler_idp_device_init(this, dof)
    class(euler_idp_device_t), intent(inout) :: this
    type(dofmap_t), target, intent(in) :: dof
    character(len=48) :: name
    integer :: component

    call this%free()
    do component = 1, EULER_IDP_NCOMP
       write(name, '(A,I0)') 'euler_idp_device_local_', component
       call this%local_residual(component)%init(dof, trim(name))
       write(name, '(A,I0)') 'euler_idp_device_candidate_', component
       call this%low_candidate(component)%init(dof, trim(name))
    end do
    call this%viscosity_sum%init(dof, 'euler_idp_device_viscosity_sum')
    call this%density_lower_bound%init(dof, 'euler_idp_device_density_lower')
    call this%density_upper_bound%init(dof, 'euler_idp_device_density_upper')
    call this%entropy_lower_bound%init(dof, 'euler_idp_device_entropy_lower')
    call this%u%init(dof, 'euler_idp_device_u')
    call this%v%init(dof, 'euler_idp_device_v')
    call this%w%init(dof, 'euler_idp_device_w')
    call this%p%init(dof, 'euler_idp_device_p')
    call this%sound_speed%init(dof, 'euler_idp_device_sound_speed')
    call this%internal_energy%init(dof, 'euler_idp_device_internal_energy')
    call this%work_1%init(dof, 'euler_idp_device_work_1')
    call this%work_2%init(dof, 'euler_idp_device_work_2')
    call this%work_3%init(dof, 'euler_idp_device_work_3')
    this%max_graph_rate = 0.0_rp
    this%max_graph_wave_speed = 0.0_rp
    this%maximum_graph_timestep = huge(1.0_rp)
    this%initialized = .true.
  end subroutine euler_idp_device_init

  subroutine euler_idp_device_init_graph(this, coef, gs, &
       relax_density_bounds, low_order_only, limit_internal_energy, &
       limit_entropy, density_bound_relaxation_factor, time_order, &
       diagnostics_level, correction_tolerance)
    class(euler_idp_device_t), intent(inout) :: this
    type(coef_t), target, intent(in) :: coef
    type(gs_t), intent(inout) :: gs
    logical, intent(in) :: relax_density_bounds, low_order_only
    logical, intent(in) :: limit_internal_energy, limit_entropy
    real(kind=rp), intent(in) :: density_bound_relaxation_factor
    real(kind=rp), intent(in) :: correction_tolerance
    integer, intent(in) :: time_order, diagnostics_level
    real(kind=rp) :: local_mass, global_mass, local_error, global_error
    character(len=48) :: name
    integer :: a(4), b(4), component, direction, edge, ierr, n

#ifndef HAVE_CUDA
    call neko_error('Euler IDP device backend currently requires CUDA')
#else
    if (.not. this%initialized) then
       call neko_error('Euler IDP device object is not initialised')
    end if
    this%relax_density_bounds = relax_density_bounds
    this%low_order_only = low_order_only
    this%limit_internal_energy = limit_internal_energy
    this%limit_entropy = limit_entropy
    this%density_bound_relaxation_factor = density_bound_relaxation_factor
    this%correction_tolerance = correction_tolerance
    this%time_order = time_order
    this%diagnostics_level = diagnostics_level
    do component = 1, EULER_IDP_NCOMP
       call this%saved_state(component)%free()
       if (time_order .eq. 3) then
          write(name, '(A,I0)') 'euler_idp_device_saved_', component
          call this%saved_state(component)%init(coef%dof, trim(name))
       end if
    end do

    call this%graph%init(coef, gs)
    this%periodic_graph = this%graph%periodic_facets_covered(coef)
    this%domain_volume = coef%volume
    if (.not. ieee_is_finite(this%domain_volume) .or. &
         this%domain_volume .le. 0.0_rp) then
       call neko_error('Euler IDP requires positive domain volume')
    end if
    local_mass = huge(1.0_rp)
    if (this%graph%mass%size() .gt. 0) then
       local_mass = minval(this%graph%mass%x)
    end if
    call MPI_Allreduce(local_mass, global_mass, 1, MPI_REAL_PRECISION, &
         MPI_MIN, NEKO_COMM, ierr)
    if (.not. ieee_is_finite(global_mass) .or. global_mass .le. 0.0_rp .or. &
         global_mass .ge. this%domain_volume) then
       call neko_error('Euler IDP density relaxation requires a valid mass')
    end if
    this%density_relaxation_mass = global_mass

    n = coef%dof%size()
    allocate(this%edge_left(this%graph%n_edges))
    allocate(this%edge_right(this%graph%n_edges))
    allocate(this%edge_direction(this%graph%n_edges))
    allocate(this%edge_coefficient(3, this%graph%n_edges))
    allocate(this%diagonal_coefficient(3, n))
    allocate(this%edge_viscosity(this%graph%n_edges))
    allocate(this%correction_flux(EULER_IDP_NCOMP, this%graph%n_edges))
    allocate(this%edge_work(this%graph%n_edges))
    allocate(this%edge_limited(this%graph%n_edges))
    allocate(this%edge_density_limited(this%graph%n_edges))
    allocate(this%edge_energy_limited(this%graph%n_edges))
    allocate(this%edge_entropy_limited(this%graph%n_edges))
    allocate(this%directional_error(EULER_IDP_NCOMP))
    do edge = 1, this%graph%n_edges
       a = this%graph%left(:,edge)
       b = this%graph%right(:,edge)
       this%edge_left(edge) = a(1) + this%graph%lx * ((a(2) - 1) + &
            this%graph%ly * ((a(3) - 1) + this%graph%lz * (a(4) - 1))) - 1
       this%edge_right(edge) = b(1) + this%graph%lx * ((b(2) - 1) + &
            this%graph%ly * ((b(3) - 1) + this%graph%lz * (b(4) - 1))) - 1
    end do
    this%edge_direction = this%graph%direction
    this%edge_coefficient = this%graph%coefficient
    this%diagonal_coefficient = reshape(this%graph%diagonal_coefficient, &
         shape(this%diagonal_coefficient))
    this%edge_viscosity = 0.0_rp
    this%correction_flux = 0.0_rp
    this%edge_work = 0.0_rp
    this%edge_limited = 0.0_rp
    this%edge_density_limited = 0.0_rp
    this%edge_energy_limited = 0.0_rp
    this%edge_entropy_limited = 0.0_rp
    this%directional_error = 0.0_rp
    call euler_idp_device_map_graph(this)

    this%work_1%x = 0.0_rp
    do edge = 1, this%graph%n_edges
       associate(left => this%graph%left(:,edge), &
            right => this%graph%right(:,edge))
         direction = this%graph%direction(edge)
         this%work_1%x(left(1),left(2),left(3),left(4)) = &
              this%work_1%x(left(1),left(2),left(3),left(4)) + &
              1.0_rp / (real(this%graph%n_directions, rp) * &
              this%graph%directional_degree(direction)%x( &
              left(1),left(2),left(3),left(4)))
         this%work_1%x(right(1),right(2),right(3),right(4)) = &
              this%work_1%x(right(1),right(2),right(3),right(4)) + &
              1.0_rp / (real(this%graph%n_directions, rp) * &
              this%graph%directional_degree(direction)%x( &
              right(1),right(2),right(3),right(4)))
       end associate
    end do
    call device_memcpy(this%work_1%x, this%work_1%x_d, n, HOST_TO_DEVICE, &
         sync = .true.)
    call gs%op(this%work_1, GS_OP_ADD)
    call device_memcpy(this%work_1%x, this%work_1%x_d, n, DEVICE_TO_HOST, &
         sync = .true.)
    local_error = 0.0_rp
    if (n .gt. 0) local_error = maxval(abs(this%work_1%x - 1.0_rp))
    call MPI_Allreduce(local_error, global_error, 1, MPI_REAL_PRECISION, &
         MPI_MAX, NEKO_COMM, ierr)
    this%limiter_weight_error = global_error
    if (global_error .gt. 128.0_rp * epsilon(1.0_rp)) then
       call neko_error('Euler IDP limiter occurrence weights do not sum to one')
    end if
#endif
  end subroutine euler_idp_device_init_graph

  subroutine euler_idp_device_map_graph(this)
    class(euler_idp_device_t), intent(inout) :: this
    integer :: n_edges, n

    n_edges = this%graph%n_edges
    n = this%graph%mass%size()
    call device_map(this%edge_left, this%edge_left_d, n_edges)
    call device_map(this%edge_right, this%edge_right_d, n_edges)
    call device_map(this%edge_direction, this%edge_direction_d, n_edges)
    call device_map(this%edge_coefficient, this%edge_coefficient_d, &
         3 * n_edges)
    call device_map(this%diagonal_coefficient, &
         this%diagonal_coefficient_d, 3 * n)
    call device_map(this%edge_viscosity, this%edge_viscosity_d, n_edges)
    call device_map(this%correction_flux, this%correction_flux_d, &
         EULER_IDP_NCOMP * n_edges)
    call device_map(this%edge_work, this%edge_work_d, n_edges)
    call device_map(this%edge_limited, this%edge_limited_d, n_edges)
    call device_map(this%edge_density_limited, &
         this%edge_density_limited_d, n_edges)
    call device_map(this%edge_energy_limited, &
         this%edge_energy_limited_d, n_edges)
    call device_map(this%edge_entropy_limited, &
         this%edge_entropy_limited_d, n_edges)
    call device_map(this%directional_error, this%directional_error_d, &
         EULER_IDP_NCOMP)
    call device_memcpy(this%edge_left, this%edge_left_d, n_edges, &
         HOST_TO_DEVICE, sync = .false.)
    call device_memcpy(this%edge_right, this%edge_right_d, n_edges, &
         HOST_TO_DEVICE, sync = .false.)
    call device_memcpy(this%edge_direction, this%edge_direction_d, n_edges, &
         HOST_TO_DEVICE, sync = .false.)
    call device_memcpy(this%edge_coefficient, this%edge_coefficient_d, &
         3 * n_edges, HOST_TO_DEVICE, sync = .false.)
    call device_memcpy(this%diagonal_coefficient, &
         this%diagonal_coefficient_d, 3 * n, HOST_TO_DEVICE, sync = .true.)
  end subroutine euler_idp_device_map_graph

  subroutine euler_idp_device_free(this)
    class(euler_idp_device_t), intent(inout) :: this
    integer :: component

    call euler_idp_device_unmap_graph(this)
    call this%graph%free()
    do component = 1, EULER_IDP_NCOMP
       call this%local_residual(component)%free()
       call this%low_candidate(component)%free()
       call this%saved_state(component)%free()
    end do
    call this%viscosity_sum%free()
    call this%density_lower_bound%free()
    call this%density_upper_bound%free()
    call this%entropy_lower_bound%free()
    call this%u%free(); call this%v%free(); call this%w%free()
    call this%p%free(); call this%sound_speed%free()
    call this%internal_energy%free()
    call this%work_1%free(); call this%work_2%free(); call this%work_3%free()
    this%initialized = .false.
    this%periodic_graph = .false.
    this%low_order_only = .false.
    this%relax_density_bounds = .false.
    this%limit_internal_energy = .true.
    this%limit_entropy = .true.
    this%density_bound_relaxation_factor = 1.0_rp
    this%correction_tolerance = 1.0e-10_rp
    this%time_order = 1
    this%diagnostics_level = EULER_IDP_DIAGNOSTICS_FULL
    this%max_graph_rate = 0.0_rp
    this%max_graph_wave_speed = 0.0_rp
    this%maximum_graph_timestep = huge(1.0_rp)
    this%maximum_floor_timestep = huge(1.0_rp)
    this%domain_volume = 0.0_rp
    this%density_relaxation_mass = 0.0_rp
    this%limiter_weight_error = 0.0_rp
  end subroutine euler_idp_device_free

  subroutine euler_idp_device_unmap_graph(this)
    class(euler_idp_device_t), intent(inout) :: this

    if (allocated(this%edge_left)) then
       call device_unmap(this%edge_left, this%edge_left_d)
       deallocate(this%edge_left)
    end if
    if (allocated(this%edge_right)) then
       call device_unmap(this%edge_right, this%edge_right_d)
       deallocate(this%edge_right)
    end if
    if (allocated(this%edge_direction)) then
       call device_unmap(this%edge_direction, this%edge_direction_d)
       deallocate(this%edge_direction)
    end if
    if (allocated(this%edge_coefficient)) then
       call device_unmap(this%edge_coefficient, this%edge_coefficient_d)
       deallocate(this%edge_coefficient)
    end if
    if (allocated(this%diagonal_coefficient)) then
       call device_unmap(this%diagonal_coefficient, &
            this%diagonal_coefficient_d)
       deallocate(this%diagonal_coefficient)
    end if
    if (allocated(this%edge_viscosity)) then
       call device_unmap(this%edge_viscosity, this%edge_viscosity_d)
       deallocate(this%edge_viscosity)
    end if
    if (allocated(this%correction_flux)) then
       call device_unmap(this%correction_flux, this%correction_flux_d)
       deallocate(this%correction_flux)
    end if
    if (allocated(this%edge_work)) then
       call device_unmap(this%edge_work, this%edge_work_d)
       deallocate(this%edge_work)
    end if
    if (allocated(this%edge_limited)) then
       call device_unmap(this%edge_limited, this%edge_limited_d)
       deallocate(this%edge_limited)
    end if
    if (allocated(this%edge_density_limited)) then
       call device_unmap(this%edge_density_limited, &
            this%edge_density_limited_d)
       deallocate(this%edge_density_limited)
    end if
    if (allocated(this%edge_energy_limited)) then
       call device_unmap(this%edge_energy_limited, &
            this%edge_energy_limited_d)
       deallocate(this%edge_energy_limited)
    end if
    if (allocated(this%edge_entropy_limited)) then
       call device_unmap(this%edge_entropy_limited, &
            this%edge_entropy_limited_d)
       deallocate(this%edge_entropy_limited)
    end if
    if (allocated(this%directional_error)) then
       call device_unmap(this%directional_error, this%directional_error_d)
       deallocate(this%directional_error)
    end if
  end subroutine euler_idp_device_unmap_graph

  subroutine euler_idp_device_primitives(this, rho, m_x, m_y, m_z, energy, &
       gamma, internal_energy_floor, label)
    class(euler_idp_device_t), intent(inout) :: this
    type(field_t), intent(in) :: rho, m_x, m_y, m_z, energy
    real(kind=rp), intent(in) :: gamma, internal_energy_floor
    character(len=*), intent(in) :: label
    integer :: n

#ifdef HAVE_CUDA
    n = rho%size()
    call cuda_euler_idp_primitives(rho%x_d, m_x%x_d, m_y%x_d, m_z%x_d, &
         energy%x_d, this%u%x_d, this%v%x_d, this%w%x_d, this%p%x_d, &
         this%sound_speed%x_d, this%internal_energy%x_d, this%work_1%x_d, &
         gamma, internal_energy_floor, n)
    if (device_glmax(this%work_1%x_d, n) .gt. 0.0_rp) then
       call neko_error('Euler IDP ' // trim(label) // &
            ' has invalid density or internal energy on the device')
    end if
#else
    call neko_error('Euler IDP device primitives require CUDA')
#endif
  end subroutine euler_idp_device_primitives

  subroutine euler_idp_device_update_graph_viscosity(this, rho, m_x, m_y, &
       m_z, energy, gs, gamma, graph_wave_speed)
    class(euler_idp_device_t), intent(inout) :: this
    type(field_t), intent(in) :: rho, m_x, m_y, m_z, energy
    type(gs_t), intent(inout) :: gs
    real(kind=rp), intent(in) :: gamma
    type(field_t), intent(in), optional :: graph_wave_speed
    type(c_ptr) :: graph_wave_speed_d
    real(kind=rp) :: min_wave_speed
    integer :: has_user, n, n_edges

#ifdef HAVE_CUDA
    n = rho%size(); n_edges = this%graph%n_edges
    has_user = 0
    graph_wave_speed_d = rho%x_d
    if (present(graph_wave_speed)) then
       has_user = 1
       graph_wave_speed_d = graph_wave_speed%x_d
    end if
    call cuda_euler_idp_graph_viscosity(rho%x_d, m_x%x_d, m_y%x_d, &
         m_z%x_d, energy%x_d, graph_wave_speed_d, has_user, &
         this%edge_left_d, this%edge_right_d, this%edge_coefficient_d, &
         this%edge_viscosity_d, this%viscosity_sum%x_d, this%edge_work_d, &
         gamma, n, n_edges)
    min_wave_speed = device_glmin(this%edge_work_d, n_edges)
    this%max_graph_wave_speed = device_glmax(this%edge_work_d, n_edges)
    if (.not. ieee_is_finite(min_wave_speed) .or. &
         .not. ieee_is_finite(this%max_graph_wave_speed) .or. &
         min_wave_speed .lt. 0.0_rp) then
       call neko_error('Euler IDP graph wave speed must be finite and nonnegative')
    end if
    call gs%op(this%viscosity_sum, GS_OP_ADD)
    call cuda_euler_idp_graph_rate(this%viscosity_sum%x_d, &
         this%graph%mass%x_d, this%work_1%x_d, n)
    this%max_graph_rate = device_glmax(this%work_1%x_d, n)
    if (this%max_graph_rate .gt. 0.0_rp) then
       this%maximum_graph_timestep = 1.0_rp / this%max_graph_rate
    else
       this%maximum_graph_timestep = huge(1.0_rp)
    end if
#else
    call neko_error('Euler IDP graph viscosity requires CUDA')
#endif
  end subroutine euler_idp_device_update_graph_viscosity

  subroutine euler_idp_device_prepare_stage(this, rho, m_x, m_y, m_z, &
       energy, gs, gamma, internal_energy_floor, primitives_valid, &
       graph_wave_speed)
    class(euler_idp_device_t), intent(inout) :: this
    type(field_t), intent(in) :: rho, m_x, m_y, m_z, energy
    type(gs_t), intent(inout) :: gs
    real(kind=rp), intent(in) :: gamma, internal_energy_floor
    logical, intent(in) :: primitives_valid
    type(field_t), intent(in), optional :: graph_wave_speed

    if (.not. primitives_valid) then
       call euler_idp_device_primitives(this, rho, m_x, m_y, m_z, energy, &
            gamma, internal_energy_floor, 'prepared stage state')
    end if
    call euler_idp_device_update_graph_viscosity(this, rho, m_x, m_y, m_z, &
         energy, gs, gamma, graph_wave_speed)
  end subroutine euler_idp_device_prepare_stage

  subroutine euler_idp_device_forward_euler(this, rho, m_x, m_y, m_z, &
       energy, coef, gs, gamma, internal_energy_floor, dt, time, diagnostics, &
       stage, entropy_viscosity_fraction, graph_wave_speed)
    class(euler_idp_device_t), intent(inout) :: this
    type(field_t), intent(in) :: rho, m_x, m_y, m_z, energy
    type(coef_t), intent(inout) :: coef
    type(gs_t), intent(inout) :: gs
    real(kind=rp), intent(in) :: gamma, internal_energy_floor, dt
    type(time_state_t), intent(in) :: time
    type(euler_idp_diagnostics_t), intent(inout) :: diagnostics
    integer, intent(in) :: stage
    type(field_t), intent(in), optional :: entropy_viscosity_fraction
    type(field_t), intent(in), optional :: graph_wave_speed
    type(c_ptr) :: entropy_fraction_d
    real(kind=rp) :: global_error(EULER_IDP_NCOMP)
    real(kind=rp) :: maximum_floor_timestep
    character(len=2 * LOG_SIZE) :: message
    integer :: affine, check_base, component, enforce_energy, enforce_entropy
    integer :: global_edge_count, global_node_count
    integer :: has_entropy, ierr, low_only, n, n_edges, periodic, scalar_mode

#ifndef HAVE_CUDA
    call neko_error('Euler IDP Forward Euler update requires CUDA')
#else
    call diagnostics%reset()
    diagnostics%stage = stage
    diagnostics%stage_time = time%t
    diagnostics%entropy_viscosity_enabled = &
         present(entropy_viscosity_fraction)
    n = rho%size(); n_edges = this%graph%n_edges
    call MPI_Allreduce(n, global_node_count, 1, MPI_INTEGER, MPI_SUM, &
         NEKO_COMM, ierr)
    call MPI_Allreduce(n_edges, global_edge_count, 1, MPI_INTEGER, MPI_SUM, &
         NEKO_COMM, ierr)
    periodic = merge(1, 0, this%periodic_graph)
    scalar_mode = merge(1, 0, present(graph_wave_speed) .and. &
         .not. this%limit_internal_energy .and. .not. this%limit_entropy)

    call cuda_euler_idp_low_residual(rho%x_d, m_x%x_d, m_y%x_d, m_z%x_d, &
         energy%x_d, this%diagonal_coefficient_d, this%edge_left_d, &
         this%edge_right_d, this%edge_coefficient_d, &
         this%edge_viscosity_d, this%low_candidate(1)%x_d, &
         this%low_candidate(2)%x_d, this%low_candidate(3)%x_d, &
         this%low_candidate(4)%x_d, this%low_candidate(5)%x_d, gamma, &
         periodic, n, n_edges)
    do component = 1, EULER_IDP_NCOMP
       call gs%op(this%low_candidate(component), GS_OP_ADD)
    end do
    call cuda_euler_idp_scale_residual(this%low_candidate(1)%x_d, &
         this%low_candidate(2)%x_d, this%low_candidate(3)%x_d, &
         this%low_candidate(4)%x_d, this%low_candidate(5)%x_d, &
         coef%Binv_d, n)

    affine = merge(1, 0, this%graph%affine)
    call cuda_euler_idp_reconstruct(rho%x_d, m_x%x_d, m_y%x_d, m_z%x_d, &
         energy%x_d, this%diagonal_coefficient_d, this%edge_left_d, &
         this%edge_right_d, this%edge_coefficient_d, &
         this%local_residual(1)%x_d, this%local_residual(2)%x_d, &
         this%local_residual(3)%x_d, this%local_residual(4)%x_d, &
         this%local_residual(5)%x_d, this%correction_flux_d, &
         this%directional_error_d, coef%B_d, coef%jacinv_d, coef%drdx_d, &
         coef%drdy_d, coef%drdz_d, coef%dsdx_d, coef%dsdy_d, coef%dsdz_d, &
         coef%dtdx_d, coef%dtdy_d, coef%dtdz_d, coef%Xh%dx_d, coef%Xh%dy_d, &
         coef%Xh%dz_d, coef%Xh%wx_d, coef%Xh%wy_d, coef%Xh%wz_d, gamma, &
         this%graph%lx, this%graph%ly, this%graph%lz, this%graph%nelv, &
         this%graph%edges_per_element, affine, n, n_edges)

    diagnostics%max_graph_wave_speed = this%max_graph_wave_speed
    diagnostics%max_graph_rate = this%max_graph_rate
    diagnostics%max_graph_cfl = dt * this%max_graph_rate
    diagnostics%min_convex_weight = 1.0_rp - diagnostics%max_graph_cfl
    if (diagnostics%max_graph_cfl .gt. 1.0_rp + &
         32.0_rp * epsilon(1.0_rp)) then
       write(message, '(A,I0,A,ES13.6,A,ES13.6)') &
            'Euler IDP SSPRK stage ', stage, &
            ' graph CFL exceeds one: ', diagnostics%max_graph_cfl, &
            ', maximum dt: ', this%maximum_graph_timestep
       call neko_error(trim(message))
    end if

    if (this%diagnostics_level .eq. EULER_IDP_DIAGNOSTICS_FULL) then
       call cuda_euler_idp_nodal_wave(this%u%x_d, this%v%x_d, this%w%x_d, &
            this%sound_speed%x_d, this%work_1%x_d, n)
       diagnostics%max_nodal_wave_speed = device_glmax(this%work_1%x_d, n)
       call device_memcpy(this%directional_error, this%directional_error_d, &
            EULER_IDP_NCOMP, DEVICE_TO_HOST, sync = .true.)
       call MPI_Allreduce(this%directional_error, global_error, &
            EULER_IDP_NCOMP, MPI_REAL_PRECISION, MPI_MAX, NEKO_COMM, ierr)
       diagnostics%reconstruction_residual = global_error
       diagnostics%correction_element_compatibility = global_error
       if (maxval(global_error) .gt. 10.0_rp * this%correction_tolerance) then
          call neko_error('Euler IDP device reconstruction is not compatible')
       end if
    end if

    if (this%limit_internal_energy) then
       call cuda_euler_idp_floor_timestep(rho%x_d, m_x%x_d, m_y%x_d, &
            m_z%x_d, energy%x_d, this%low_candidate(1)%x_d, &
            this%low_candidate(2)%x_d, this%low_candidate(3)%x_d, &
            this%low_candidate(4)%x_d, this%low_candidate(5)%x_d, &
            this%work_1%x_d, internal_energy_floor, &
            this%maximum_graph_timestep, n)
       maximum_floor_timestep = device_glmin(this%work_1%x_d, n)
    else
       maximum_floor_timestep = this%maximum_graph_timestep
    end if
    this%maximum_floor_timestep = maximum_floor_timestep
    diagnostics%maximum_floor_timestep = maximum_floor_timestep
    if (this%limit_internal_energy .and. dt .gt. &
         maximum_floor_timestep * (1.0_rp + 32.0_rp * epsilon(1.0_rp))) then
       write(message, '(A,I0,A,ES13.6,A,ES13.6)') &
            'Euler IDP SSPRK stage ', stage, &
            ' timestep exceeds the floor-aware limit: ', dt, &
            ', maximum dt: ', maximum_floor_timestep
       call neko_error(trim(message))
    end if
    call cuda_euler_idp_low_update(rho%x_d, m_x%x_d, m_y%x_d, m_z%x_d, &
         energy%x_d, this%low_candidate(1)%x_d, &
         this%low_candidate(2)%x_d, this%low_candidate(3)%x_d, &
         this%low_candidate(4)%x_d, this%low_candidate(5)%x_d, dt, &
         scalar_mode, n)

    call euler_idp_device_compute_bounds(this, rho, m_x, m_y, m_z, energy, &
         gs, gamma)
    if (this%diagnostics_level .eq. EULER_IDP_DIAGNOSTICS_FULL) then
       call euler_idp_device_check_low_bounds(this, gamma)
    end if

    has_entropy = merge(1, 0, present(entropy_viscosity_fraction))
    low_only = merge(1, 0, this%low_order_only)
    entropy_fraction_d = rho%x_d
    if (present(entropy_viscosity_fraction)) then
       entropy_fraction_d = entropy_viscosity_fraction%x_d
       diagnostics%max_entropy_viscosity = &
            device_glmax(entropy_viscosity_fraction%x_d, n)
       diagnostics%mean_entropy_viscosity = &
            device_glsum(entropy_viscosity_fraction%x_d, n) / &
            real(max(1, global_node_count), rp)
    end if
    call cuda_euler_idp_blend(rho%x_d, m_x%x_d, m_y%x_d, m_z%x_d, &
         energy%x_d, this%edge_left_d, this%edge_right_d, &
         this%edge_viscosity_d, entropy_fraction_d, this%correction_flux_d, &
         dt, has_entropy, low_only, scalar_mode, n_edges)

    if (scalar_mode .eq. 0) then
       call euler_idp_device_primitives(this, this%low_candidate(1), &
            this%low_candidate(2), this%low_candidate(3), &
            this%low_candidate(4), this%low_candidate(5), gamma, &
            internal_energy_floor, 'low-order candidate')
    else if (device_glmin(this%low_candidate(1)%x_d, n) .le. 0.0_rp) then
       call neko_error('Scalar IDP low-order candidate has invalid density')
    end if

    enforce_energy = merge(1, 0, this%limit_internal_energy)
    enforce_entropy = merge(1, 0, this%limit_entropy)
    check_base = merge(1, 0, &
         this%diagnostics_level .ne. EULER_IDP_DIAGNOSTICS_OFF)
    call cuda_euler_idp_limiter(this%low_candidate(1)%x_d, &
         this%low_candidate(2)%x_d, this%low_candidate(3)%x_d, &
         this%low_candidate(4)%x_d, this%low_candidate(5)%x_d, &
         this%density_lower_bound%x_d, this%density_upper_bound%x_d, &
         this%entropy_lower_bound%x_d, this%graph%mass%x_d, &
         this%graph%directional_degree(1)%x_d, &
         this%graph%directional_degree(2)%x_d, &
         this%graph%directional_degree(3)%x_d, this%edge_left_d, &
         this%edge_right_d, this%edge_direction_d, this%correction_flux_d, &
         this%edge_work_d, this%edge_limited_d, &
         this%edge_density_limited_d, this%edge_energy_limited_d, &
         this%edge_entropy_limited_d, gamma, internal_energy_floor, &
         enforce_energy, enforce_entropy, check_base, &
         this%graph%n_directions, n_edges)

    diagnostics%limiter_weight_error = this%limiter_weight_error
    if (this%diagnostics_level .eq. EULER_IDP_DIAGNOSTICS_FULL) then
       diagnostics%min_limiter = device_glmin(this%edge_work_d, n_edges)
       diagnostics%max_limiter = device_glmax(this%edge_work_d, n_edges)
       diagnostics%mean_limiter = device_glsum(this%edge_work_d, n_edges) / &
            real(max(1, global_edge_count), rp)
       diagnostics%limited_edge_fraction = &
            device_glsum(this%edge_limited_d, n_edges) / &
            real(max(1, global_edge_count), rp)
       diagnostics%density_limited_edges = nint( &
            device_glsum(this%edge_density_limited_d, n_edges))
       diagnostics%internal_energy_limited_edges = nint( &
            device_glsum(this%edge_energy_limited_d, n_edges))
       diagnostics%entropy_limited_edges = nint( &
            device_glsum(this%edge_entropy_limited_d, n_edges))
    end if

    call cuda_euler_idp_incidence(this%edge_left_d, this%edge_right_d, &
         this%correction_flux_d, this%low_candidate(1)%x_d, &
         this%low_candidate(2)%x_d, this%low_candidate(3)%x_d, &
         this%low_candidate(4)%x_d, this%low_candidate(5)%x_d, &
         this%local_residual(1)%x_d, this%local_residual(2)%x_d, &
         this%local_residual(3)%x_d, this%local_residual(4)%x_d, &
         this%local_residual(5)%x_d, n, n_edges)
    do component = 1, EULER_IDP_NCOMP
       call gs%op(this%local_residual(component), GS_OP_ADD)
    end do
    call cuda_euler_idp_correction_update(this%low_candidate(1)%x_d, &
         this%low_candidate(2)%x_d, this%low_candidate(3)%x_d, &
         this%low_candidate(4)%x_d, this%low_candidate(5)%x_d, &
         this%local_residual(1)%x_d, this%local_residual(2)%x_d, &
         this%local_residual(3)%x_d, this%local_residual(4)%x_d, &
         this%local_residual(5)%x_d, this%graph%mass%x_d, n)
#endif
  end subroutine euler_idp_device_forward_euler

  subroutine euler_idp_device_compute_bounds(this, rho, m_x, m_y, m_z, &
       energy, gs, gamma)
    class(euler_idp_device_t), intent(inout) :: this
    type(field_t), intent(in) :: rho, m_x, m_y, m_z, energy
    type(gs_t), intent(inout) :: gs
    real(kind=rp), intent(in) :: gamma
    integer :: limit_entropy, n, n_edges, relax

#ifdef HAVE_CUDA
    n = rho%size(); n_edges = this%graph%n_edges
    relax = merge(1, 0, this%relax_density_bounds)
    limit_entropy = merge(1, 0, this%limit_entropy)
    call cuda_euler_idp_bounds_init(rho%x_d, m_x%x_d, m_y%x_d, m_z%x_d, &
         energy%x_d, this%density_lower_bound%x_d, &
         this%density_upper_bound%x_d, this%entropy_lower_bound%x_d, &
         this%work_1%x_d, gamma, limit_entropy, n)
    if (this%limit_entropy) then
       call device_copy(this%sound_speed%x_d, this%entropy_lower_bound%x_d, n)
    end if
    call cuda_euler_idp_bounds_edges(rho%x_d, m_x%x_d, m_y%x_d, m_z%x_d, &
         this%edge_left_d, this%edge_right_d, this%edge_direction_d, &
         this%edge_coefficient_d, this%edge_viscosity_d, &
         this%graph%directional_degree(1)%x_d, &
         this%graph%directional_degree(2)%x_d, &
         this%graph%directional_degree(3)%x_d, &
         this%density_lower_bound%x_d, this%density_upper_bound%x_d, &
         this%sound_speed%x_d, this%entropy_lower_bound%x_d, &
         this%work_1%x_d, relax, limit_entropy, n_edges)
    call gs%op(this%density_lower_bound, GS_OP_MIN)
    call gs%op(this%density_upper_bound, GS_OP_MAX)
    if (this%limit_entropy) call gs%op(this%entropy_lower_bound, GS_OP_MIN)
    if (this%relax_density_bounds) then
       call gs%op(this%work_1, GS_OP_ADD)
       call cuda_euler_idp_relax_edges(this%edge_left_d, this%edge_right_d, &
            this%edge_direction_d, this%graph%directional_degree(1)%x_d, &
            this%graph%directional_degree(2)%x_d, &
            this%graph%directional_degree(3)%x_d, this%work_1%x_d, &
            this%work_2%x_d, n, n_edges)
       call gs%op(this%work_2, GS_OP_ADD)
       call cuda_euler_idp_relax_finalize(this%density_lower_bound%x_d, &
            this%density_upper_bound%x_d, this%work_2%x_d, &
            this%density_bound_relaxation_factor, &
            this%density_relaxation_mass, this%domain_volume, &
            this%graph%n_directions, n)
    end if
#endif
  end subroutine euler_idp_device_compute_bounds

  subroutine euler_idp_device_check_low_bounds(this, gamma)
    class(euler_idp_device_t), intent(inout) :: this
    real(kind=rp), intent(in) :: gamma
    real(kind=rp) :: scale, violation
    integer :: limit_entropy, n

#ifdef HAVE_CUDA
    n = this%low_candidate(1)%size()
    limit_entropy = merge(1, 0, this%limit_entropy)
    call cuda_euler_idp_validate(this%low_candidate(1)%x_d, &
         this%low_candidate(2)%x_d, this%low_candidate(3)%x_d, &
         this%low_candidate(4)%x_d, this%low_candidate(5)%x_d, &
         this%density_lower_bound%x_d, this%density_upper_bound%x_d, &
         this%entropy_lower_bound%x_d, this%work_1%x_d, this%work_2%x_d, &
         this%work_3%x_d, this%sound_speed%x_d, gamma, limit_entropy, n)
    violation = max(device_glmax(this%work_1%x_d, n), &
         device_glmax(this%work_2%x_d, n))
    scale = max(1.0_rp, device_glmax(this%density_upper_bound%x_d, n), &
         device_glmax(this%low_candidate(1)%x_d, n))
    if (violation .gt. 256.0_rp * epsilon(1.0_rp) * scale) then
       call neko_error('Euler IDP low-order density is outside its bounds')
    end if
    if (this%limit_entropy .and. &
         device_glmax(this%work_3%x_d, n) .gt. 0.0_rp) then
       call neko_error('Euler IDP low-order state violates its entropy bound')
    end if
#endif
  end subroutine euler_idp_device_check_low_bounds

  subroutine euler_idp_device_apply_boundary_conditions(this, rho, m_x, &
       m_y, m_z, energy, density_bcs, velocity_bcs, pressure_bcs, gamma, &
       internal_energy_floor, time, label, refresh)
    class(euler_idp_device_t), intent(inout) :: this
    type(field_t), intent(inout) :: rho, m_x, m_y, m_z, energy
    type(bc_list_t), intent(inout) :: density_bcs, velocity_bcs, pressure_bcs
    real(kind=rp), intent(in) :: gamma, internal_energy_floor
    type(time_state_t), intent(in), optional :: time
    character(len=*), intent(in) :: label
    logical, intent(in), optional :: refresh
    class(bc_t), pointer :: boundary
    logical :: refresh_
    integer :: i, n

#ifndef HAVE_CUDA
    call neko_error('Euler IDP boundary conditions require CUDA')
#else
    refresh_ = .false.
    if (present(refresh)) refresh_ = refresh
    if (refresh_) then
       do i = 1, density_bcs%size()
          boundary => density_bcs%get(i); boundary%updated = .false.
       end do
       do i = 1, velocity_bcs%size()
          boundary => velocity_bcs%get(i); boundary%updated = .false.
       end do
       do i = 1, pressure_bcs%size()
          boundary => pressure_bcs%get(i); boundary%updated = .false.
       end do
       nullify(boundary)
    end if
    n = rho%size()
    if (present(time)) then
       call density_bcs%apply(rho, time = time, strong = .true.)
    else
       call density_bcs%apply(rho, strong = .true.)
    end if
    call cuda_euler_idp_update_uvw(this%u%x_d, this%v%x_d, this%w%x_d, &
         m_x%x_d, m_y%x_d, m_z%x_d, rho%x_d, n)
    if (present(time)) then
       call velocity_bcs%apply(this%u, this%v, this%w, time = time, &
            strong = .true.)
    else
       call velocity_bcs%apply(this%u, this%v, this%w, strong = .true.)
    end if
    call cuda_euler_idp_update_momentum_pressure(m_x%x_d, m_y%x_d, &
         m_z%x_d, this%p%x_d, this%internal_energy%x_d, this%u%x_d, &
         this%v%x_d, this%w%x_d, energy%x_d, rho%x_d, gamma, n)
    if (present(time)) then
       call pressure_bcs%apply(this%p, time = time, strong = .true.)
    else
       call pressure_bcs%apply(this%p, strong = .true.)
    end if
    call cuda_euler_idp_update_energy(energy%x_d, this%p%x_d, &
         this%internal_energy%x_d, gamma, internal_energy_floor, n)
    call euler_idp_device_primitives(this, rho, m_x, m_y, m_z, energy, &
         gamma, internal_energy_floor, label)
#endif
  end subroutine euler_idp_device_apply_boundary_conditions

  subroutine euler_idp_device_apply_candidate_boundary(this, density_bcs, &
       velocity_bcs, pressure_bcs, gamma, internal_energy_floor, time, label)
    class(euler_idp_device_t), intent(inout) :: this
    type(bc_list_t), intent(inout) :: density_bcs, velocity_bcs, pressure_bcs
    real(kind=rp), intent(in) :: gamma, internal_energy_floor
    type(time_state_t), intent(in) :: time
    character(len=*), intent(in) :: label

    call this%apply_boundary_conditions(this%low_candidate(1), &
         this%low_candidate(2), this%low_candidate(3), &
         this%low_candidate(4), this%low_candidate(5), density_bcs, &
         velocity_bcs, pressure_bcs, gamma, internal_energy_floor, time, label)
  end subroutine euler_idp_device_apply_candidate_boundary

  subroutine euler_idp_device_observe_candidate(this, gs, gamma, &
       internal_energy_floor, time, label, primitives_valid, observation, &
       graph_wave_speed)
    class(euler_idp_device_t), intent(inout) :: this
    type(gs_t), intent(inout) :: gs
    real(kind=rp), intent(in) :: gamma, internal_energy_floor
    type(time_state_t), intent(in) :: time
    character(len=*), intent(in) :: label
    logical, intent(in) :: primitives_valid
    type(euler_idp_state_observation_t), intent(out) :: observation
    type(field_t), intent(in), optional :: graph_wave_speed
    integer :: n

#ifdef HAVE_CUDA
    if (.not. primitives_valid) then
       call euler_idp_device_primitives(this, this%low_candidate(1), &
            this%low_candidate(2), this%low_candidate(3), &
            this%low_candidate(4), this%low_candidate(5), gamma, &
            internal_energy_floor, label)
    end if
    n = this%low_candidate(1)%size()
    observation = euler_idp_state_observation_t()
    observation%time = time%t
    observation%min_density = &
         device_glmin(this%low_candidate(1)%x_d, n)
    observation%min_internal_energy = &
         device_glmin(this%internal_energy%x_d, n)
    observation%min_pressure = device_glmin(this%p%x_d, n)
    call cuda_euler_idp_nodal_wave(this%u%x_d, this%v%x_d, this%w%x_d, &
         this%sound_speed%x_d, this%work_1%x_d, n)
    observation%max_nodal_wave_speed = device_glmax(this%work_1%x_d, n)
    call euler_idp_device_update_graph_viscosity(this, &
         this%low_candidate(1), this%low_candidate(2), &
         this%low_candidate(3), this%low_candidate(4), &
         this%low_candidate(5), gs, gamma, graph_wave_speed)
    observation%max_graph_wave_speed = this%max_graph_wave_speed
    observation%max_graph_rate = this%max_graph_rate
#else
    call neko_error('Euler IDP candidate observation requires CUDA')
#endif
  end subroutine euler_idp_device_observe_candidate

  subroutine euler_idp_device_validate_candidate(this, gamma, diagnostics, &
       label)
    class(euler_idp_device_t), intent(inout) :: this
    real(kind=rp), intent(in) :: gamma
    type(euler_idp_diagnostics_t), intent(inout) :: diagnostics
    character(len=*), intent(in) :: label
    real(kind=rp) :: scale
    integer :: limit_entropy, n

    if (this%diagnostics_level .eq. EULER_IDP_DIAGNOSTICS_OFF) return
#ifdef HAVE_CUDA
    n = this%low_candidate(1)%size()
    limit_entropy = merge(1, 0, this%limit_entropy)
    call cuda_euler_idp_validate(this%low_candidate(1)%x_d, &
         this%low_candidate(2)%x_d, this%low_candidate(3)%x_d, &
         this%low_candidate(4)%x_d, this%low_candidate(5)%x_d, &
         this%density_lower_bound%x_d, this%density_upper_bound%x_d, &
         this%entropy_lower_bound%x_d, this%work_1%x_d, this%work_2%x_d, &
         this%work_3%x_d, this%sound_speed%x_d, gamma, limit_entropy, n)
    diagnostics%max_density_lower_violation = &
         device_glmax(this%work_1%x_d, n)
    diagnostics%max_density_upper_violation = &
         device_glmax(this%work_2%x_d, n)
    scale = max(1.0_rp, device_glmax(this%density_upper_bound%x_d, n), &
         device_glmax(this%low_candidate(1)%x_d, n))
    if (max(diagnostics%max_density_lower_violation, &
         diagnostics%max_density_upper_violation) .gt. &
         512.0_rp * epsilon(1.0_rp) * scale) then
       call neko_error('Euler IDP ' // trim(label) // &
            ' density violates its local bounds')
    end if
    if (this%limit_entropy) then
       diagnostics%max_entropy_lower_violation = &
            device_glmax(this%work_3%x_d, n)
       if (diagnostics%max_entropy_lower_violation .gt. 0.0_rp) then
          call neko_error('Euler IDP ' // trim(label) // &
               ' violates its local entropy bound')
       end if
    end if
    diagnostics%min_density = device_glmin(this%low_candidate(1)%x_d, n)
    diagnostics%min_internal_energy = &
         device_glmin(this%internal_energy%x_d, n)
    diagnostics%min_pressure = device_glmin(this%p%x_d, n)
    diagnostics%min_specific_entropy = &
         device_glmin(this%sound_speed%x_d, n)
#else
    call neko_error('Euler IDP candidate validation requires CUDA')
#endif
  end subroutine euler_idp_device_validate_candidate

  subroutine euler_idp_device_save_state(this, rho, m_x, m_y, m_z, energy)
    class(euler_idp_device_t), intent(inout) :: this
    type(field_t), intent(in) :: rho, m_x, m_y, m_z, energy
    integer :: n

    if (this%time_order .ne. 3) then
       call neko_error('Euler IDP device saved state is not allocated')
    end if
    n = rho%size()
    call device_copy(this%saved_state(1)%x_d, rho%x_d, n)
    call device_copy(this%saved_state(2)%x_d, m_x%x_d, n)
    call device_copy(this%saved_state(3)%x_d, m_y%x_d, n)
    call device_copy(this%saved_state(4)%x_d, m_z%x_d, n)
    call device_copy(this%saved_state(5)%x_d, energy%x_d, n)
  end subroutine euler_idp_device_save_state

  subroutine euler_idp_device_combine_stage(this, rho, m_x, m_y, m_z, &
       energy, saved_weight, candidate_weight)
    class(euler_idp_device_t), intent(in) :: this
    type(field_t), intent(inout) :: rho, m_x, m_y, m_z, energy
    real(kind=rp), intent(in) :: saved_weight, candidate_weight
    integer :: n

    n = rho%size()
    if (saved_weight .eq. 0.0_rp) then
       call device_cmult2(rho%x_d, this%low_candidate(1)%x_d, &
            candidate_weight, n)
       call device_cmult2(m_x%x_d, this%low_candidate(2)%x_d, &
            candidate_weight, n)
       call device_cmult2(m_y%x_d, this%low_candidate(3)%x_d, &
            candidate_weight, n)
       call device_cmult2(m_z%x_d, this%low_candidate(4)%x_d, &
            candidate_weight, n)
       call device_cmult2(energy%x_d, this%low_candidate(5)%x_d, &
            candidate_weight, n)
    else
       call device_add3s2(rho%x_d, this%saved_state(1)%x_d, &
            this%low_candidate(1)%x_d, saved_weight, candidate_weight, n)
       call device_add3s2(m_x%x_d, this%saved_state(2)%x_d, &
            this%low_candidate(2)%x_d, saved_weight, candidate_weight, n)
       call device_add3s2(m_y%x_d, this%saved_state(3)%x_d, &
            this%low_candidate(3)%x_d, saved_weight, candidate_weight, n)
       call device_add3s2(m_z%x_d, this%saved_state(4)%x_d, &
            this%low_candidate(4)%x_d, saved_weight, candidate_weight, n)
       call device_add3s2(energy%x_d, this%saved_state(5)%x_d, &
            this%low_candidate(5)%x_d, saved_weight, candidate_weight, n)
    end if
  end subroutine euler_idp_device_combine_stage

  subroutine euler_idp_device_copy_primitives(this, u, v, w, p)
    class(euler_idp_device_t), intent(in) :: this
    type(field_t), intent(inout) :: u, v, w, p
    integer :: n

    n = u%size()
    call device_copy(u%x_d, this%u%x_d, n)
    call device_copy(v%x_d, this%v%x_d, n)
    call device_copy(w%x_d, this%w%x_d, n)
    call device_copy(p%x_d, this%p%x_d, n)
  end subroutine euler_idp_device_copy_primitives

end module euler_idp_device
