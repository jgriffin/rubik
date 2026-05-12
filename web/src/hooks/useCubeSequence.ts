// React hook wrapping the framework-agnostic `createCubeSequence`
// factory in `state/cubeSequence.ts`. Bridges the controller's
// subscribe-based update model onto React via `useSyncExternalStore`.
//
// Snapshot strategy: version counter.
//   - The hook subscribes to the sequence and increments an internal
//     version counter on every change notification.
//   - `useSyncExternalStore`'s `getSnapshot` returns the counter.
//   - The hook returns the live `CubeSequence` ref (stable across the
//     hook's lifetime, with getter properties that always reflect
//     current state).
// React re-renders when the counter changes; the caller reads getters
// on the returned ref. This keeps the ref stable for downstream
// `useEffect` deps, avoids snapshot-identity churn that would trigger
// "snapshot is not stable" warnings, and matches the controller's own
// "live getters" idiom.
//
// Caller idiom (stable spec):
//   const spec = useMemo(
//     () => ({ startFacelet, moves, msPerMove }),
//     [startFacelet, moves.join(" "), msPerMove],
//   );
//   const seq = useCubeSequence(spec);
//
// Block D's cross-surface scrubber sync will skip this hook and lift
// the factory call into a context provider, then subscribe from
// multiple components — same factory, same contract, no fork.

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useSyncExternalStore,
} from "react";
import {
  createCubeSequence,
  type CubeSequence,
  type CubeSequenceSpec,
} from "../state/cubeSequence";

export type { CubeSequence, CubeSequenceSpec, CubeSequenceStatus }
  from "../state/cubeSequence";

export function useCubeSequence(spec: CubeSequenceSpec): CubeSequence {
  // Build the sequence once per (referentially-stable) spec. The
  // caller is responsible for memoizing `spec` so identity-equality
  // stays meaningful — see header doc.
  const sequence = useMemo(() => createCubeSequence(spec), [spec]);

  // Version counter that bumps on every state-change notification.
  // Held in a ref so it survives re-renders; mutated synchronously
  // inside the subscribe callback so `useSyncExternalStore` picks up
  // the change on its next read.
  const versionRef = useRef(0);

  const subscribe = useCallback(
    (onStoreChange: () => void): (() => void) => {
      return sequence.subscribe(() => {
        versionRef.current += 1;
        onStoreChange();
      });
    },
    [sequence],
  );

  // getSnapshot must be referentially stable across renders for the
  // same store (otherwise React re-subscribes). It returns a primitive
  // (the counter value) so identity equality is just === on numbers.
  // The ref is stable across the hook's lifetime; the closure has no
  // captured deps and is stable too.
  const getSnapshot = useCallback(() => versionRef.current, []);

  // Server snapshot: identical for SSR. The factory never starts an
  // rAF loop during SSR (no `window` access at module level), so the
  // initial version 0 is the correct SSR snapshot. The caller should
  // not rely on the sequence playing under SSR.
  const getServerSnapshot = useCallback((): number => 0, []);

  useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  // Cleanup on unmount or sequence-replacement: pause the outgoing
  // sequence so its rAF callback doesn't fire after we've already
  // dropped the React subscription. (The factory itself cancels its
  // own rAF on `pause()`; this is the defensive belt-and-suspenders
  // for unmount during playback.)
  useEffect(() => {
    return () => {
      sequence.pause();
    };
  }, [sequence]);

  return sequence;
}
