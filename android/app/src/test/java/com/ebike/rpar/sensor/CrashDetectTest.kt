package com.ebike.rpar.sensor

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CrashDetectTest {
    @Test
    fun gravityIsNotACrash() {
        assertFalse(severeImpact(0.0, 0.0, 9.81))
    }

    @Test
    fun potholeTwoGIsNotACrash() {
        assertFalse(severeImpact(0.0, 0.0, 9.81 + 20.0))
    }

    @Test
    fun fourGResidualIsACrash() {
        assertTrue(severeImpact(0.0, 0.0, 9.81 + 40.0))
    }

    @Test
    fun violentSpinIsACrash() {
        assertTrue(severeImpact(0.0, 0.0, 9.81, gyroRadS = 16.0))
    }

    @Test
    fun residualMatchesMagnitudeMinusGravity() {
        assertEquals(0.0, linearAccelResidual(0.0, 0.0, 9.81), 1e-6)
        assertEquals(40.19, linearAccelResidual(0.0, 0.0, 50.0), 0.01)
    }
}
