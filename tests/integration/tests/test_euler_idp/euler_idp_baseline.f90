module user
  use neko
  implicit none

contains

  subroutine user_setup(user)
    type(user_t), intent(inout) :: user

    user%initial_conditions => initial_conditions
    user%finalize => finalize
  end subroutine user_setup

  subroutine initial_conditions(scheme_name, fields)
    character(len=*), intent(in) :: scheme_name
    type(field_list_t), intent(inout) :: fields
    type(field_t), pointer :: rho, u, v, w, p
    real(kind=rp) :: x, y, initial_density, initial_pressure
    integer :: i

    rho => fields%get_by_name("fluid_rho")
    u => fields%get_by_name("u")
    v => fields%get_by_name("v")
    w => fields%get_by_name("w")
    p => fields%get_by_name("p")
    initial_density = neko_const_registry%get_real_scalar("initial_density")
    initial_pressure = neko_const_registry%get_real_scalar("initial_pressure")

    do i = 1, rho%dof%size()
       x = rho%dof%x(i, 1, 1, 1)
       y = rho%dof%y(i, 1, 1, 1)
       rho%x(i, 1, 1, 1) = initial_density + &
            0.2_rp * sin(2.0_rp * pi * (x + y))
       u%x(i, 1, 1, 1) = 2.5_rp
       v%x(i, 1, 1, 1) = -0.5_rp
       w%x(i, 1, 1, 1) = 0.0_rp
       p%x(i, 1, 1, 1) = initial_pressure
    end do
  end subroutine initial_conditions

  subroutine finalize(time)
    type(time_state_t), intent(in) :: time
    type(coef_t), pointer :: coef
    type(field_t), pointer :: rho, m_x, m_y, m_z, energy
    character(len=1024) :: dump_file
    real(kind=rp) :: values(8)
    integer :: n, dump_status, dump_unit

    coef => neko_user_access%case%fluid%c_Xh
    rho => neko_registry%get_field("fluid_rho")
    m_x => neko_registry%get_field("m_x")
    m_y => neko_registry%get_field("m_y")
    m_z => neko_registry%get_field("m_z")
    energy => neko_registry%get_field("E")
    n = rho%size()

    values(1) = glsc2(rho%x, coef%B, n)
    values(2) = glsc2(m_x%x, coef%B, n)
    values(3) = glsc2(m_y%x, coef%B, n)
    values(4) = glsc2(m_z%x, coef%B, n)
    values(5) = glsc2(energy%x, coef%B, n)
    values(6) = sqrt(glsc3(rho%x, rho%x, coef%B, n))
    values(7) = glmin(rho%x, n)
    values(8) = glmax(rho%x, n)

    if (pe_rank .eq. 0) then
       write(*, '(A,I0,8(1X,ES25.16E3))') &
            "EULER_IDP_BASELINE ", time%tstep, values

       ! The integration test compares every conserved degree of freedom.
       call get_environment_variable("NEKO_TEST_IDP_DUMP", dump_file, &
            status=dump_status)
       if (dump_status .eq. 0 .and. len_trim(dump_file) .gt. 0) then
          open(newunit=dump_unit, file=trim(dump_file), access="stream", &
               form="unformatted", status="replace", action="write")
          write(dump_unit) rho%x, m_x%x, m_y%x, m_z%x, energy%x
          close(dump_unit)
       end if
    end if
  end subroutine finalize

end module user
