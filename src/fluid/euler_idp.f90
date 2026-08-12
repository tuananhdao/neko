! Copyright (c) 2026, The Neko Authors
! All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions
! are met:
!
!   * Redistributions of source code must retain the above copyright
!     notice, this list of conditions and the following disclaimer.
!
!   * Redistributions in binary form must reproduce the above
!     copyright notice, this list of conditions and the following
!     disclaimer in the documentation and/or other materials provided
!     with the distribution.
!
!   * Neither the name of the authors nor the names of its
!     contributors may be used to endorse or promote products derived
!     from this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
! "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
! LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
! FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
! COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
! INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
! BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
! LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
! LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
! ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
! POSSIBILITY OF SUCH DAMAGE.
!
!> Configuration and ownership for the Euler GLL IDP method.
module euler_idp
  use json_module, only : json_file
  use json_utils, only : json_get
  implicit none
  private

  !> Top-level owner for the Euler IDP path.
  type, public :: euler_idp_t
     logical :: enabled = .false.
     ! IDP in progress: Add persistent graph, stage, and work data here.
   contains
     procedure, pass(this) :: init => euler_idp_init
     procedure, pass(this) :: free => euler_idp_free
  end type euler_idp_t

contains

  !> Initialize the top-level Euler IDP owner.
  !> @param this Euler IDP owner.
  !> @param params Case parameters.
  subroutine euler_idp_init(this, params)
    class(euler_idp_t), intent(inout) :: this
    type(json_file), intent(inout) :: params

    call this%free()
    if (.not. params%valid_path("case.numerics.euler_idp.enabled")) return
    call json_get(params, "case.numerics.euler_idp.enabled", this%enabled)
  end subroutine euler_idp_init

  !> Reset the top-level Euler IDP owner.
  !> @param this Euler IDP owner.
  subroutine euler_idp_free(this)
    class(euler_idp_t), intent(inout) :: this

    this%enabled = .false.
  end subroutine euler_idp_free

end module euler_idp
