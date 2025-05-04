import unittest
from turbine import Turbine

class TestTurbine(unittest.TestCase):

    def test_initialization(self):
        turbine = Turbine("TestTurbine", 1500, 3.5, 25.0)
        self.assertEqual(turbine.name, "TestTurbine")
        self.assertEqual(turbine.capacity_kw, 1500)
        self.assertEqual(turbine.cut_in_speed, 3.5)
        self.assertEqual(turbine.cut_out_speed, 25.0)

    def test_is_operational_true(self):
        turbine = Turbine("TestTurbine", 1500, 3.5, 25.0)
        self.assertTrue(turbine.is_operational(10.0))

    def test_is_operational_false_low_speed(self):
        turbine = Turbine("TestTurbine", 1500, 3.5, 25.0)
        self.assertFalse(turbine.is_operational(2.0))

    def test_is_operational_false_high_speed(self):
        turbine = Turbine("TestTurbine", 1500, 3.5, 25.0)
        self.assertFalse(turbine.is_operational(30.0))

if __name__ == '__main__':
    unittest.main()