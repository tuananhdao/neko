! Integration-test driver for the periodic Euler IDP cases.
module user
  use, intrinsic :: ieee_arithmetic, only : ieee_is_finite
  use neko
  use fluid_scheme_compressible_ns, only : fluid_scheme_compressible_ns_t
  implicit none

  character(len=:), allocatable :: problem
  integer :: polynomial_order = 0
  real(kind=rp) :: gamma_ref = 1.4_rp
  real(kind=rp) :: energy_floor = 1.0e-12_rp
  real(kind=rp) :: initial_integrals(5) = 0.0_rp
  real(kind=rp) :: minimum_stage_density = huge(1.0_rp)
  real(kind=rp) :: minimum_stage_internal_energy = huge(1.0_rp)
  real(kind=rp) :: minimum_limiter = 1.0_rp
  real(kind=rp) :: maximum_limited_fraction = 0.0_rp
  real(kind=rp) :: maximum_graph_cfl = 0.0_rp
  real(kind=rp) :: maximum_density_lower_violation = 0.0_rp
  real(kind=rp) :: maximum_density_upper_violation = 0.0_rp
  real(kind=rp) :: maximum_entropy_lower_violation = 0.0_rp
  real(kind=rp) :: maximum_stage_conservation(5) = 0.0_rp
  integer :: stage_count = 0
  logical :: all_stages_finite = .true.

contains

  !> Register the test callbacks.
  subroutine user_setup(user)
    type(user_t), intent(inout) :: user

    user%startup => startup
    user%initial_conditions => initial_conditions
    user%initialize => initialize
    user%compute => monitor_stages
    user%finalize => finalize
  end subroutine user_setup

  !> Read test-only settings from the generated case file.
  subroutine startup(params)
    type(json_file), intent(inout) :: params

    call json_get(params, 'test.problem', problem)
    call json_get(params, 'case.numerics.polynomial_order', &
         polynomial_order)
    call json_get(params, 'case.fluid.gamma', gamma_ref)
    call json_get(params, &
         'case.numerics.euler_idp.internal_energy_floor', energy_floor)
  end subroutine startup

  !> Set one of the four periodic Euler states exercised by pytest.
  subroutine initial_conditions(scheme_name, fields)
    character(len=*), intent(in) :: scheme_name
    type(field_list_t), intent(inout) :: fields
    type(field_t), pointer :: rho, u, v, w, p
    real(kind=rp) :: x, y
    integer :: i

    if (trim(scheme_name) .ne. 'fluid') return

    rho => fields%get_by_name('fluid_rho')
    u => fields%get_by_name('u')
    v => fields%get_by_name('v')
    w => fields%get_by_name('w')
    p => fields%get_by_name('p')

    do i = 1, rho%size()
       x = rho%dof%x(i, 1, 1, 1)
       y = rho%dof%y(i, 1, 1, 1)

       select case (trim(problem))
       case ('free_stream')
          rho%x(i, 1, 1, 1) = 1.0_rp
          u%x(i, 1, 1, 1) = 0.7_rp
          v%x(i, 1, 1, 1) = -0.2_rp
          w%x(i, 1, 1, 1) = 0.1_rp
          p%x(i, 1, 1, 1) = 1.0_rp
       case ('smooth_transport')
          rho%x(i, 1, 1, 1) = 1.0_rp + &
               0.2_rp * sin(2.0_rp * pi * (x + y))
          u%x(i, 1, 1, 1) = 2.5_rp
          v%x(i, 1, 1, 1) = -0.5_rp
          w%x(i, 1, 1, 1) = 0.0_rp
          p%x(i, 1, 1, 1) = 1.0_rp
       case ('periodic_discontinuity')
          u%x(i, 1, 1, 1) = 0.0_rp
          v%x(i, 1, 1, 1) = 0.0_rp
          w%x(i, 1, 1, 1) = 0.0_rp
          if (x .lt. 0.25_rp .or. x .ge. 0.75_rp) then
             rho%x(i, 1, 1, 1) = 1.0_rp
             p%x(i, 1, 1, 1) = 1.0_rp
          else
             rho%x(i, 1, 1, 1) = 0.125_rp
             p%x(i, 1, 1, 1) = 0.1_rp
          end if
       case ('near_vacuum')
          u%x(i, 1, 1, 1) = 0.0_rp
          v%x(i, 1, 1, 1) = 0.0_rp
          w%x(i, 1, 1, 1) = 0.0_rp
          if (x .lt. 0.25_rp .or. x .ge. 0.75_rp) then
             rho%x(i, 1, 1, 1) = 1.0_rp
             p%x(i, 1, 1, 1) = &
                  (gamma_ref - 1.0_rp) * 1.0e-1_rp
          else
             rho%x(i, 1, 1, 1) = 1.0e-3_rp
             p%x(i, 1, 1, 1) = &
                  (gamma_ref - 1.0_rp) * 1.0e-10_rp
          end if
       case default
          call neko_error('Unknown Euler IDP integration-test problem')
       end select
    end do
  end subroutine initial_conditions

  !> Save the five initial global conserved integrals.
  subroutine initialize(time)
    type(time_state_t), intent(in) :: time
    type(coef_t), pointer :: coef
    type(field_t), pointer :: rho, m_x, m_y, m_z, energy

    coef => neko_user_access%case%fluid%c_Xh
    rho => neko_registry%get_field('fluid_rho')
    m_x => neko_registry%get_field('m_x')
    m_y => neko_registry%get_field('m_y')
    m_z => neko_registry%get_field('m_z')
    energy => neko_registry%get_field('E')

    initial_integrals(1) = glsc2(rho%x, coef%B, rho%size())
    initial_integrals(2) = glsc2(m_x%x, coef%B, rho%size())
    initial_integrals(3) = glsc2(m_y%x, coef%B, rho%size())
    initial_integrals(4) = glsc2(m_z%x, coef%B, rho%size())
    initial_integrals(5) = glsc2(energy%x, coef%B, rho%size())
  end subroutine initialize

  !> Reduce the diagnostics over every SSPRK3 stage and time step.
  subroutine monitor_stages(time)
    type(time_state_t), intent(in) :: time
    integer :: stage

    select type (fluid => neko_user_access%case%fluid)
    type is (fluid_scheme_compressible_ns_t)
       do stage = 1, 3
          associate(diagnostic => &
               fluid%euler_idp_solver%stage_diagnostics(stage))
            stage_count = stage_count + 1
            minimum_stage_density = min(minimum_stage_density, &
                 diagnostic%min_density)
            minimum_stage_internal_energy = &
                 min(minimum_stage_internal_energy, &
                 diagnostic%min_internal_energy)
            minimum_limiter = min(minimum_limiter, &
                 diagnostic%min_limiter)
            maximum_limited_fraction = max(maximum_limited_fraction, &
                 diagnostic%limited_edge_fraction)
            maximum_graph_cfl = max(maximum_graph_cfl, &
                 diagnostic%max_graph_cfl)
            maximum_density_lower_violation = &
                 max(maximum_density_lower_violation, &
                 diagnostic%max_density_lower_violation)
            maximum_density_upper_violation = &
                 max(maximum_density_upper_violation, &
                 diagnostic%max_density_upper_violation)
            maximum_entropy_lower_violation = &
                 max(maximum_entropy_lower_violation, &
                 diagnostic%max_entropy_lower_violation)
            maximum_stage_conservation = max(maximum_stage_conservation, &
                 abs(diagnostic%limited_conservation))
            all_stages_finite = all_stages_finite .and. &
                 diagnostic_is_finite(diagnostic%min_density, &
                 diagnostic%min_internal_energy, &
                 diagnostic%min_limiter, &
                 diagnostic%limited_edge_fraction, &
                 diagnostic%max_graph_cfl, &
                 diagnostic%max_density_lower_violation, &
                 diagnostic%max_density_upper_violation, &
                 diagnostic%max_entropy_lower_violation, &
                 diagnostic%limited_conservation)
          end associate
       end do
    class default
       call neko_error('Euler IDP test requires the compressible scheme')
    end select
  end subroutine monitor_stages

  !> Report exact errors, conservation drift, and aggregated diagnostics.
  subroutine finalize(time)
    type(time_state_t), intent(in) :: time
    type(coef_t), pointer :: coef
    type(field_t), pointer :: rho, m_x, m_y, m_z, energy
    real(kind=rp), allocatable :: work(:)
    real(kind=rp) :: state(5), exact(5), final_integrals(5)
    real(kind=rp) :: errors(5), drifts(5)
    real(kind=rp) :: minimum_density, minimum_internal_energy
    integer :: component, i, n

    coef => neko_user_access%case%fluid%c_Xh
    rho => neko_registry%get_field('fluid_rho')
    m_x => neko_registry%get_field('m_x')
    m_y => neko_registry%get_field('m_y')
    m_z => neko_registry%get_field('m_z')
    energy => neko_registry%get_field('E')
    n = rho%size()
    allocate(work(n))

    final_integrals(1) = glsc2(rho%x, coef%B, n)
    final_integrals(2) = glsc2(m_x%x, coef%B, n)
    final_integrals(3) = glsc2(m_y%x, coef%B, n)
    final_integrals(4) = glsc2(m_z%x, coef%B, n)
    final_integrals(5) = glsc2(energy%x, coef%B, n)
    drifts = abs(final_integrals - initial_integrals)

    do i = 1, n
       work(i) = rho%x(i, 1, 1, 1)
    end do
    minimum_density = glmin(work, n)
    do i = 1, n
       work(i) = energy%x(i, 1, 1, 1) - 0.5_rp * &
            (m_x%x(i, 1, 1, 1)**2 + m_y%x(i, 1, 1, 1)**2 + &
            m_z%x(i, 1, 1, 1)**2) / rho%x(i, 1, 1, 1)
    end do
    minimum_internal_energy = glmin(work, n)

    errors = 0.0_rp
    if (trim(problem) .eq. 'free_stream' .or. &
         trim(problem) .eq. 'smooth_transport') then
       do component = 1, 5
          do i = 1, n
             state = [rho%x(i, 1, 1, 1), m_x%x(i, 1, 1, 1), &
                  m_y%x(i, 1, 1, 1), m_z%x(i, 1, 1, 1), &
                  energy%x(i, 1, 1, 1)]
             call exact_state(rho%dof%x(i, 1, 1, 1), &
                  rho%dof%y(i, 1, 1, 1), time%t, exact)
             work(i) = abs(state(component) - exact(component))
          end do
          errors(component) = glmax(work, n)
       end do
    end if

    all_stages_finite = all_stages_finite .and. &
         fields_are_finite(rho, m_x, m_y, m_z, energy)

    if (pe_rank .eq. 0) then
       write(*, '(A,1X,A,4(1X,I0),1X,L1)') 'EULER_IDP_RESULT', &
            trim(problem), polynomial_order, pe_size, time%tstep, &
            stage_count, all_stages_finite
       write(*, '(A,5(1X,ES25.16E3))') 'EULER_IDP_ERROR', errors
       write(*, '(A,5(1X,ES25.16E3))') 'EULER_IDP_DRIFT', drifts
       write(*, '(A,3(1X,ES25.16E3))') 'EULER_IDP_LIMITER', &
            minimum_limiter, maximum_limited_fraction, maximum_graph_cfl
       write(*, '(A,4(1X,ES25.16E3))') 'EULER_IDP_STATE', &
            minimum_stage_density, minimum_stage_internal_energy, &
            minimum_density, minimum_internal_energy
       write(*, '(A,3(1X,ES25.16E3))') 'EULER_IDP_BOUNDS', &
            maximum_density_lower_violation, &
            maximum_density_upper_violation, &
            maximum_entropy_lower_violation
       write(*, '(A,5(1X,ES25.16E3))') 'EULER_IDP_CONSERVATION', &
            maximum_stage_conservation
    end if

    deallocate(work)
  end subroutine finalize

  !> Exact conserved state for the constant and smooth transport cases.
  subroutine exact_state(x, y, time, state)
    real(kind=rp), intent(in) :: x, y, time
    real(kind=rp), intent(out) :: state(5)
    real(kind=rp) :: density, velocity(3), pressure

    pressure = 1.0_rp
    if (trim(problem) .eq. 'free_stream') then
       density = 1.0_rp
       velocity = [0.7_rp, -0.2_rp, 0.1_rp]
    else
       density = 1.0_rp + 0.2_rp * sin(2.0_rp * pi * &
            (x + y - 2.0_rp * time))
       velocity = [2.5_rp, -0.5_rp, 0.0_rp]
    end if

    state(1) = density
    state(2:4) = density * velocity
    state(5) = pressure / (gamma_ref - 1.0_rp) + &
         0.5_rp * density * dot_product(velocity, velocity)
  end subroutine exact_state

  !> Check every scalar carried by one stage diagnostic.
  logical function diagnostic_is_finite(rho_min, energy_min, limiter_min, &
       limited_fraction, graph_cfl, density_lower_violation, &
       density_upper_violation, entropy_violation, conservation) &
       result(finite)
    real(kind=rp), intent(in) :: rho_min, energy_min, limiter_min
    real(kind=rp), intent(in) :: limited_fraction, graph_cfl
    real(kind=rp), intent(in) :: density_lower_violation
    real(kind=rp), intent(in) :: density_upper_violation, entropy_violation
    real(kind=rp), intent(in) :: conservation(5)

    finite = ieee_is_finite(rho_min) .and. &
         ieee_is_finite(energy_min) .and. &
         ieee_is_finite(limiter_min) .and. &
         ieee_is_finite(limited_fraction) .and. &
         ieee_is_finite(graph_cfl) .and. &
         ieee_is_finite(density_lower_violation) .and. &
         ieee_is_finite(density_upper_violation) .and. &
         ieee_is_finite(entropy_violation) .and. &
         all(ieee_is_finite(conservation))
  end function diagnostic_is_finite

  !> Check the five final conserved fields for NaN or infinity.
  logical function fields_are_finite(rho, m_x, m_y, m_z, energy) &
       result(finite)
    type(field_t), intent(in) :: rho, m_x, m_y, m_z, energy

    finite = all(ieee_is_finite(rho%x)) .and. &
         all(ieee_is_finite(m_x%x)) .and. &
         all(ieee_is_finite(m_y%x)) .and. &
         all(ieee_is_finite(m_z%x)) .and. &
         all(ieee_is_finite(energy%x))
  end function fields_are_finite

end module user
