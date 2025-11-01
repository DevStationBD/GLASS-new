#!/usr/bin/env python3
"""
Serial Port Listener - Test Program for GLASS Defect Notifications

This program listens to a serial port and displays received commands in the terminal.
Use this to test the GLASS inference system's serial defect notifications.
"""

import serial
import time
import argparse
import sys
from datetime import datetime

def listen_to_serial(port, baud_rate, timeout=1.0):
    """Listen to serial port and display received commands"""
    
    print(f"🔌 Attempting to connect to serial port: {port}")
    print(f"⚙️  Baud Rate: {baud_rate}")
    print(f"⏱️  Timeout: {timeout}s")
    print("-" * 50)
    
    try:
        # Open serial connection
        ser = serial.Serial(
            port=port,
            baudrate=baud_rate,
            timeout=timeout,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE
        )
        
        print(f"✅ Connected to {port} successfully!")
        print(f"📡 Listening for commands... (Press Ctrl+C to stop)")
        print("=" * 50)
        
        command_count = 0
        
        while True:
            try:
                # Read data from serial port
                if ser.in_waiting > 0:
                    # Read line (until newline character)
                    raw_data = ser.readline()
                    
                    # Decode bytes to string
                    try:
                        command = raw_data.decode('utf-8').strip()
                        if command:  # Only process non-empty commands
                            command_count += 1
                            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                            
                            print(f"[{timestamp}] Command #{command_count}: '{command}'")
                            
                            # Parse if it's a GLASS defect notification
                            if command == 'stc':
                                print(f"  🎯 GLASS Defect Alert: Simple trigger command")
                            else:
                                print(f"  📨 Generic Command: {command}")
                            
                    except UnicodeDecodeError:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Received non-UTF8 data: {raw_data}")
                
                else:
                    # Small delay to prevent busy waiting
                    time.sleep(0.01)
                    
            except KeyboardInterrupt:
                print(f"\n🛑 Stopping serial listener...")
                break
            except Exception as e:
                print(f"❌ Error reading from serial port: {e}")
                break
        
        # Close connection
        ser.close()
        print(f"📡 Serial connection closed")
        print(f"📊 Total commands received: {command_count}")
        
    except serial.SerialException as e:
        print(f"❌ Failed to connect to serial port {port}: {e}")
        print(f"💡 Make sure:")
        print(f"   - The port {port} exists and is available")
        print(f"   - You have permission to access the port")
        print(f"   - No other program is using the port")
        print(f"   - The baud rate ({baud_rate}) matches the sender")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Serial Port Listener for GLASS Defect Notifications')
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0',
                       help='Serial port to listen on (default: /dev/ttyUSB0)')
    parser.add_argument('--baud_rate', type=int, default=115200,
                       help='Baud rate (default: 115200)')
    parser.add_argument('--timeout', type=float, default=1.0,
                       help='Read timeout in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    print("🔍 GLASS Serial Defect Notification Listener")
    print("=" * 50)
    
    # Start listening
    success = listen_to_serial(args.port, args.baud_rate, args.timeout)
    
    if not success:
        print("\n💡 Common serial ports to try:")
        print("   Linux: /dev/ttyUSB0, /dev/ttyUSB1, /dev/ttyACM0")
        print("   Windows: COM1, COM2, COM3, COM4")
        print("   macOS: /dev/cu.usbserial-*, /dev/cu.usbmodem-*")
        sys.exit(1)

if __name__ == '__main__':
    main()
