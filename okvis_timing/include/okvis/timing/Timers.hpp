#ifndef TIMERS_H
#define TIMERS_H

#include <okvis/timing/Timer.hpp>

namespace okvis {
#ifdef DEACTIVATE_TIMERS
  typedef okvis::timing::DummyTimer TimerSwitchable;
#else
  typedef okvis::timing::Timer TimerSwitchable;
#endif

#define TimerRefStruct9(NAME, TIMER1, TIMER2, TIMER3, TIMER4, TIMER5, TIMER6,   \
                       TIMER7, TIMER8, TIMER9)                                 \
  struct NAME {                                                                \
    okvis::TimerSwitchable &TIMER1;                                                   \
    okvis::TimerSwitchable &TIMER2;                                                   \
    okvis::TimerSwitchable &TIMER3;                                                   \
    okvis::TimerSwitchable &TIMER4;                                                   \
    okvis::TimerSwitchable &TIMER5;                                                   \
    okvis::TimerSwitchable &TIMER6;                                                   \
    okvis::TimerSwitchable &TIMER7;                                                   \
    okvis::TimerSwitchable &TIMER8;                                                   \
    okvis::TimerSwitchable &TIMER9;                                                   \
    NAME(okvis::TimerSwitchable &T1, okvis::TimerSwitchable &T2, okvis::TimerSwitchable &T3,        \
         okvis::TimerSwitchable &T4, okvis::TimerSwitchable &T5, okvis::TimerSwitchable &T6,        \
         okvis::TimerSwitchable &T7, okvis::TimerSwitchable &T8, okvis::TimerSwitchable &T9)        \
        : TIMER1(T1), TIMER2(T2), TIMER3(T3), TIMER4(T4), TIMER5(T5),          \
          TIMER6(T6), TIMER7(T7), TIMER8(T8), TIMER9(T9) {}                    \
  };

#define TimerRefStruct3(NAME, TIMER1, TIMER2, TIMER3)                          \
  struct NAME {                                                                \
    okvis::TimerSwitchable &TIMER1;                                                   \
    okvis::TimerSwitchable &TIMER2;                                                   \
    okvis::TimerSwitchable &TIMER3;                                                   \
    NAME(okvis::TimerSwitchable &T1, okvis::TimerSwitchable &T2, okvis::TimerSwitchable &T3)        \
        : TIMER1(T1), TIMER2(T2), TIMER3(T3) {}                                \
  };

#define TimerRefStruct4(NAME, TIMER1, TIMER2, TIMER3, TIMER4)                  \
  struct NAME {                                                                \
    okvis::TimerSwitchable &TIMER1;                                                   \
    okvis::TimerSwitchable &TIMER2;                                                   \
    okvis::TimerSwitchable &TIMER3;                                                   \
    okvis::TimerSwitchable &TIMER4;                                                   \
    NAME(okvis::TimerSwitchable &T1, okvis::TimerSwitchable &T2, okvis::TimerSwitchable &T3,        \
         okvis::TimerSwitchable &T4)                                                  \
        : TIMER1(T1), TIMER2(T2), TIMER3(T3), TIMER4(T4) {}                    \
  };

}  // namespace okvis

#endif // TIMERS_H
