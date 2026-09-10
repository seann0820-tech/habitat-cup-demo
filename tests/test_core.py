import unittest
import numpy as np
from cup_baseline.metrics import n85, summarize, continual_metrics
from cup_baseline.model import Ensemble, MPC, prior


class CoreTests(unittest.TestCase):
    def test_real_ensemble_checkpoint_includes_meta_and_rejects_wrong_agent(self):
        from cup_baseline.metacognition import MetaMonitor, FEATURES, ACTIONS
        import copy
        model=Ensemble(7,mode='learned');model.monitor=MetaMonitor(7)
        o=dict(x=np.array([0,0,0,.5,.9,0],np.float32),joints=np.zeros(7,np.float32),
               rays=np.full(12,3,np.float32),phase=0)
        for i in range(40):
            action=np.array([.5,0,0,0,0],np.float32)
            y=prior(o['x'][None],action[None])[0]
            model.add(o,action,dict(x=y))
        # Use public episode supervision, including real calibration holdouts.
        # This fixture is synthetic and never emitted as an experiment result.
        for episode in range(15):
            trace=[]
            for step in range(16):
                trace.append(dict(features=np.zeros(len(FEATURES)).tolist(),
                    choice=step%len(ACTIONS),step=step,phase=0,reward=.1,discount=.99,
                    allowed=[True]*len(ACTIONS),diagnostic_valid=True,
                    phase_success=episode%2,collision=(episode+1)%2,prediction_error=.1))
            model.monitor.add_episode(trace,episode%2,16)
        model.fit(updates=1)
        self.assertGreater(model.monitor.train_updates,0)
        state=copy.deepcopy(model.snapshot());other=Ensemble(8,mode='learned');other.monitor=MetaMonitor(8)
        other.restore(state)
        for a,b in zip(model.monitor.weights,other.monitor.weights):np.testing.assert_array_equal(a,b)
        for a,b in zip(model.monitor.target_weights,other.monitor.target_weights):np.testing.assert_array_equal(a,b)
        self.assertEqual(model.monitor.calibration,other.monitor.calibration)
        self.assertEqual(len(model.monitor.samples),len(other.monitor.samples))
        self.assertEqual(len(model.monitor.calibration_samples),len(other.monitor.calibration_samples))
        with self.assertRaises(ValueError):Ensemble(9,mode='learned').restore(state)
        with self.assertRaises(ValueError):other.restore(Ensemble(9,mode='learned').snapshot())

    def test_n85_confirmation_and_censor(self):
        curve = [dict(train_episodes=i*10,train_steps=i*100,success_rate=v)
                 for i,v in enumerate([.9,.8,.85,.9])]
        self.assertEqual(n85(curve,30)['episodes'],20)
        self.assertFalse(n85(curve[:3],20)['reached'])
        self.assertIsNone(n85(curve[:3],20)['episodes'])

    def test_forgetting(self):
        r = [[.8,.1,.1],[.6,.9,.2],[.5,.8,.95]]
        m = continual_metrics(r)
        self.assertAlmostEqual(m['average_forgetting'],.2)
        self.assertAlmostEqual(m['backward_transfer'],-.2)

    def test_planner_accounting_and_eval_freeze(self):
        model = Ensemble(3,hidden=16)
        planner = MPC(model,horizon=3,candidates=12,iterations=2,elite=3)
        o = dict(x=np.array([0,0,0,.5,.9,0],np.float32),joints=np.zeros(7,np.float32),
                 rays=np.full(12,3,np.float32),phase=0,nav_goal=np.array([2,0]),
                 cup=np.array([2.7,.8,0]),ee_goal=np.array([2.7,.8,0]),obstacles=np.empty((0,2)))
        before = [p.detach().clone() for p in model.nets.parameters()]
        a = planner.act(o)
        self.assertTrue(np.all(a[2:]==0))
        self.assertEqual(model.count.nn_forward_calls,3*3*2)
        self.assertEqual(model.count.nn_sample_forwards,3*3*2*12)
        self.assertEqual(model.count.nn_flops_est,model.flops_one*3*3*2*12)
        self.assertEqual(len(model.inputs),0)
        self.assertTrue(all(np.array_equal(x.numpy(),y.detach().numpy()) for x,y in zip(before,model.nets.parameters())))

    def test_learning_changes_model_on_unseen_transitions(self):
        # Synthetic controller unit check only; these are never reported as Habitat results.
        model = Ensemble(4,hidden=32)
        rng = np.random.default_rng(7)
        for i in range(200):
            x = np.array([0,0,0,.5,.9,0],np.float32)
            a = rng.uniform(-1,1,5).astype(np.float32); a[2:]=0
            o=dict(x=x,joints=np.zeros(7,np.float32),rays=np.full(12,3,np.float32),phase=0)
            y=prior(x[None],a[None])[0]; y[0]*=.65
            model.add(o,a,dict(x=y))
        x=np.tile(np.array([0,0,0,.5,.9,0],np.float32),(20,1))
        a=rng.uniform(-1,1,(20,5)).astype(np.float32);a[:,2:]=0
        truth=prior(x,a);truth[:,0]*=.65
        before=model.predict(x,a,o)[0]
        model.fit(updates=150,batch=64)
        after=model.predict(x,a,o)[0]
        self.assertLess(np.mean((after[:,0]-truth[:,0])**2),np.mean((before[:,0]-truth[:,0])**2))

    def test_learned_mode_does_not_use_dynamics_prior_at_evaluation(self):
        model = Ensemble(1,hidden=16,mode='learned')
        x=np.array([[0,0,0,.5,.9,0]],np.float32)
        a=np.array([[1,0,0,0,0]],np.float32)
        o=dict(joints=np.zeros(7,np.float32),rays=np.full(12,3,np.float32),phase=0)
        y,_=model.predict(x,a,o)
        np.testing.assert_allclose(y,x)
        model.collect_with_prior=True
        y,_=model.predict(x,a,o)
        self.assertGreater(y[0,0],0)
        # Warm-up must not secretly change how supervised transition targets form.
        obs=dict(x=x[0],**o)
        model.add(obs,a[0],dict(x=y[0]))
        self.assertGreater(model.targets[0][0],.9)


if __name__ == '__main__':
    unittest.main()
